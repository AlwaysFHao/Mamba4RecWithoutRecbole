# -*- coding: utf-8 -*-
# @Author : Hao Fan
# @Time : 2024/5/27

import torch
import torch.nn as nn
from mamba_ssm import Mamba
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
import math

from einops import repeat, rearrange

"""
Refer from Mamba4Rec, Mamba-Py and Mamba
Mamba4Rec: https://github.com/chengkai-liu/Mamba4Rec
Mamba-Py: https://github.com/alxndrTL/mamba.py
Mamba: https://github.com/state-spaces/mamba
"""

# Try to import causal_conv1d
# If this fails, nn.Conv1d add padding is used instead
try:
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
except ImportError:
    causal_conv1d_fn, causal_conv1d_update = None, None


class Mamba4Rec(nn.Module):
    def __init__(self,
                 items_num,
                 hidden_size,
                 d_state,
                 d_conv,
                 expand,
                 num_layers,
                 dropout_prob, ):
        super(Mamba4Rec, self).__init__()
        self.num_layers = num_layers
        self.item_embedding = nn.Embedding(items_num + 1, hidden_size, padding_idx=0)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(dropout_prob)

        self.mamba_layers = nn.ModuleList([
            MambaLayer(
                d_model=hidden_size,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                dropout=dropout_prob,
                num_layers=num_layers
            ) for _ in range(num_layers)
        ])

        self.loss_fct = nn.CrossEntropyLoss()

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def gather_indexes(self, output, gather_index):
        """Gathers the vectors at the specific positions over a minibatch"""
        gather_index = gather_index.view(-1, 1, 1).expand(-1, -1, output.shape[-1])
        output_tensor = output.gather(dim=1, index=gather_index)
        return output_tensor.squeeze(1)

    def forward(self, item_seq, item_seq_len):
        item_emb = self.item_embedding(item_seq)
        item_emb = self.dropout(item_emb)
        item_emb = self.LayerNorm(item_emb)

        for i in range(self.num_layers):
            item_emb = self.mamba_layers[i](item_emb)

        seq_output = self.gather_indexes(item_emb, item_seq_len - 1)
        return seq_output

    def calculate_loss(self, item_seq, item_seq_len, labels):
        seq_output = self.forward(item_seq, item_seq_len)

        test_item_emb = self.item_embedding.weight
        logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))
        loss = self.loss_fct(logits, labels)
        return loss

    def full_sort_predict(self, item_seq, item_seq_len):
        seq_output = self.forward(item_seq, item_seq_len)
        test_items_emb = self.item_embedding.weight
        scores = torch.matmul(
            seq_output, test_items_emb.transpose(0, 1)
        )
        return scores


class MambaLayer(nn.Module):
    def __init__(self, d_model, d_state, d_conv, expand, dropout, num_layers):
        """
        A single-layer MambaLayer, containing a MambaBlock and an FFN
        :param d_model: vector embedding dimension
        :param d_state: the B, C matrix dimension in Mamba
        :param d_conv: causal-conv1d kernel size
        :param expand: coefficient of expanding
        :param dropout: dropout_radio
        :param num_layers: The number of MambaLayer layers,
        used to determine whether the MambaBlock needs residuals connections
        """
        super(MambaLayer, self).__init__()
        self.num_layers = num_layers
        self.mamba = MambaBlock(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand
        )
        self.dropout = nn.Dropout(dropout)
        self.LayerNorm = nn.LayerNorm(d_model, eps=1e-12)
        self.ffn = FeedForward(
            d_model=d_model,
            inner_size=d_conv * 4,
            dropout=dropout)

    def forward(self, x):
        """
        x -> mamba(x) -> ffn(x)
        :param x: shape [batch_size, seq_len, d_model]
        :return: shape [batch_size, seq_len, d_model]
        """
        hidden = self.mamba(x)
        # Determine whether MambaBlock needs residuals by num_layers
        if self.num_layers == 1:
            hidden = self.LayerNorm(self.dropout(hidden))
        else:
            hidden = self.LayerNorm(self.dropout(hidden) + x)
        return self.ffn(hidden)


class MambaBlock(nn.Module):
    def __init__(self,
                 d_model: int,
                 d_state: int = 16,
                 d_conv: int = 4,
                 expand: int = 2,
                 dt_rank: str = "auto",
                 dt_min: float = 0.001,
                 dt_max: float = 0.1,
                 dt_init: str = "random",
                 dt_scale: float = 1.0,
                 bias: bool = False,
                 conv_bias: bool = False, ):
        """
        MambaBlock
        :param d_model: input vector embedding dimension
        :param d_state: the dimension of B and C
        :param d_conv: causal_conv1d kernel size
        :param expand: coefficient of expansion
        :param dt_rank: dimension of dt(Delta), if "auto" -> dt_rank = ceil(d_model / 16)
        :param dt_min: the min value of dt value
        :param dt_max: the max value of dt value
        :param dt_init: the initialization method of dt
        :param dt_scale: the initialization variance scaling range of dt_proj
        :param bias: whether bias is added to the linear layer
        :param conv_bias: whether bias is added to the conv layer
        """
        super(MambaBlock, self).__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        # Transform the input into input x and bypass z
        # [batch_size, seq_len, d_model] -> [batch_size, seq_len, d_inner * 2]
        self.in_proj = nn.Linear(in_features=self.d_model, out_features=self.d_inner * 2, bias=bias)

        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
        )

        # casual-conv1d required activation function parameters
        self.activation = "silu"
        self.act = nn.SiLU()

        # For generating dt(\Delta), B and C at the same time
        # [batch_size, seq_len, d_inner] -> [batch_size, seq_len, dt_rank + d_state * 2]
        self.x_proj = nn.Linear(in_features=self.d_inner,
                                out_features=self.dt_rank + self.d_state * 2,
                                bias=False)
        # dt projection
        self.dt_proj = nn.Linear(in_features=self.dt_rank, out_features=self.d_inner)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = self.dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError(f'{dt_init} initialization method not implemented')

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.randn(self.d_inner) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        ).clamp(min=dt_min)

        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        # Avoid initializing the current bias to 0 during subsequent initialization
        self.dt_proj.bias._no_reinit = True

        # S4D real initialization
        # HiPPO matrix and low-rank decomposition
        A = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32),
            "n -> d n",
            d=self.d_inner
        ).contiguous()
        A_log = torch.log(A)  # why store A in log ? to keep A < 0 (cf -torch.exp(...)) ? for gradient stability ?
        # Convert to learnable parameters
        self.A_log = nn.Parameter(A_log)
        # Set not to add regular entries
        self.A_log._no_weight_decay = True
        # D bypass parameter
        self.D = nn.Parameter(torch.ones(self.d_inner))
        # Set not to add regular entries
        self.D._no_weight_decay = True

        # output projection
        self.out_proj = nn.Linear(in_features=self.d_inner,
                                  out_features=self.d_model,
                                  bias=bias)

    def forward(self, x):
        """
        :param x: [batch_size, seq_len, d_model]
        :return: same shape [batch_size, seq_len, d_model]
        """
        # Get input_tensor shape
        batch_size, seq_len, d_model = x.shape

        # Here, the weighted summation and bias of the linear layer transformation of in_proj are treated separately
        # matmul and transpose BLH -> HBL at the same time
        xz = rearrange(
            self.in_proj.weight @ rearrange(x, "b l d -> d (b l)"),
            "d (b l) -> b d l",
            l=seq_len
        )

        # Add bias if the linear layer is biased
        if self.in_proj.bias is not None:
            xz = xz + rearrange(self.in_proj.bias.to(dtype=xz.dtype), "d -> d 1")

        # (d_inner, d_state)
        A = -torch.exp(self.A_log.float())

        # [batch_size, seq_len, d_inner * 2] -> [batch_size, seq_len, d_inner] * 2
        x, z = xz.chunk(2, dim=1)

        if causal_conv1d_fn is not None:
            x = self.act(self.conv1d(x)[..., :seq_len])
        else:
            assert self.activation in ["silu", "swish"], \
                f'{self.activation} is not supported in causal-conv1d, please use "silu" or "swish"'
            x = causal_conv1d_fn(
                x=x,
                weight=rearrange(self.conv1d.weight, "d 1 w -> d w"),
                bias=self.conv1d.bias,
                activation=self.activation
            )

        # Mamba want dt to have d as the slowest moving dimension
        # and L as the fastest moving dimension, since those are what the ssm_scan kernel expects

        # [d, batch_size, seq_len] -> [batch_size * seq_len, d]
        # d: dt_rank + 2 * d_state
        x_dbl = self.x_proj(rearrange(x, "b d l -> (b l) d"))
        # dt: [batch_size * seq_len, dt_rank]
        # B: [batch_size * seq_len, dt_state]
        # C: [batch_size * seq_len, dt_state]
        dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        # [dt_inner, dt_rank] @ [batch_size * seq_len, dt_rank]^T
        dt = self.dt_proj.weight @ dt.t()
        dt = rearrange(dt, "d (b l) -> b d l", l=seq_len)
        B = rearrange(B, "(b l) dstate -> b dstate l", l=seq_len).contiguous()
        C = rearrange(C, "(b l) dstate -> b dstate l", l=seq_len).contiguous()

        # h(t) = e^{dt @ A} h(t - 1) + e^{dt @ A - I} A^{-1} B x(t)
        # y(t) = C h(t) + D x(t)
        y = selective_scan_fn(
            u=x, delta=dt, A=A, B=B, C=C, D=self.D.float(), z=z,
            delta_bias=self.dt_proj.bias.float(), delta_softplus=True
        )

        # [batch_size, d_state, seq_len] -> [batch_size, seq_len, d_state]
        y = rearrange(y, "b d l -> b l d")
        # [batch_size, seq_len, d_state] -> [batch_size, seq_len, d_model]
        out = self.out_proj(y)
        return out


def gelu(x):
    """Implementation of the gelu activation function.

        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results)::

            0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

        Also see https://arxiv.org/abs/1606.08415
        """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


class FeedForward(nn.Module):
    def __init__(self, d_model, inner_size, dropout=0.2):
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, inner_size)
        self.fc2 = nn.Linear(inner_size, d_model)
        self.dropout = nn.Dropout(dropout)
        self.gelu = gelu
        self.LayerNorm = nn.LayerNorm(d_model, eps=1e-12)

    def forward(self, x):
        hidden = self.gelu(self.fc1(x))
        hidden = self.dropout(hidden)

        hidden = self.fc2(hidden)
        hidden = self.dropout(hidden + x)
        return hidden


if __name__ == '__main__':
    model = MambaLayer(
        # This module uses roughly 3 * expand * d_model^2 parameters
        d_model=64,
        d_state=256,
        d_conv=4,
        expand=2,
        dropout=0.2,
        num_layers=1
    ).to(torch.device('cuda'))
    input_tensor = torch.randn(2, 10, 64).to(torch.device('cuda'))
    out_tensor = model(input_tensor)
    print(out_tensor.shape)
