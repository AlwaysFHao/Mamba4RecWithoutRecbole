# -*- coding: utf-8 -*-            
# @Author : Hao Fan
# @Time : 2024/5/27

import os

import numpy
import torch
import yaml
from tqdm import tqdm

from dataloader import AmazonDataset
from mamba4rec import Mamba4Rec
from utils import setting_logging, get_full_sort_score

if __name__ == '__main__':
    torch.manual_seed(2024)
    # 定义配置项
    # 配置文件路径
    config_yaml_file_path = os.path.join('config.yaml')
    # 从配置文件加载配置
    with open(config_yaml_file_path, 'r') as stream:
        config = yaml.safe_load(stream)

    if torch.cuda.is_available() is False:
        raise EnvironmentError('没有可用GPU，Mamba当前仅支持CUDA运行！')
    device = torch.device("cuda")

    hidden_size = config['hidden_size']
    num_layers = config['num_layers']
    dropout_prob = config['dropout_prob']
    d_state = config['d_state']
    d_conv = config['d_conv']
    expand = config['expand']
    root_path = config['root_path']
    max_len = config['MAX_ITEM_LIST_LENGTH']
    logger_name = config['logger_name']
    Epoch = config['epochs']
    batch_size = config['batch_size']
    lr = config['learning_rate']
    weight_decay = config['weight_decay']
    top_k = config['top_k']
    model_saved_path = config['model_saved_path']
    model_saved_name = config['model_saved_name']

    logger = setting_logging(logger_name)

    # 新建k-core文件夹
    if not os.path.exists(model_saved_path):
        os.makedirs(model_saved_path)

    test_dataset = AmazonDataset(root_path=root_path, max_len=max_len, split='test')
    user_num = test_dataset.user_num
    item_num = test_dataset.item_num
    model = Mamba4Rec(
        items_num=item_num,
        hidden_size=hidden_size,
        d_state=d_state,
        d_conv=d_conv,
        expand=expand,
        num_layers=num_layers,
        dropout_prob=dropout_prob
    )

    # 加载模型参数
    if os.path.exists(os.path.join(model_saved_path, model_saved_name)):
        saved_state_dict = torch.load(os.path.join(model_saved_path, model_saved_name))
        max_val_hit = saved_state_dict['max_val_hit']
        saved_state_dict.pop('max_val_hit')
        model.load_state_dict(saved_state_dict)
    else:
        max_val_hit = -1.0

    model = model.to(device)
    test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    test_epoch_step = len(test_dataset) // batch_size
    total_hit = 0.0
    total_ndcg = 0.0

    model.eval()
    pbar_val = tqdm(total=test_epoch_step, desc=f'Test',
                    postfix={'hit': '?', 'ndcg': '?', 'hit_mean': '?', 'ndcg_mean': '?'},
                    mininterval=0.3)
    for index, (_, seq, label, length, _, _) in enumerate(test_dataloader):
        # 舍弃不足batch_size的批次
        if index >= test_epoch_step:
            break
        seq = seq.to(device)
        length = length.to(device)
        with torch.no_grad():
            # 模型正向传播
            score = model.full_sort_predict(
                item_seq=seq,
                item_seq_len=length
            )
            label = label.numpy()
            label = label[:, numpy.newaxis]
            hit, ndcg = get_full_sort_score(answers=label, pred_list=score, topk=top_k)

        total_hit += hit
        total_ndcg += ndcg

        pbar_val.set_postfix(**{'hit': hit, 'ndcg': ndcg,
                                'hit_mean': total_hit / (index + 1), 'ndcg_mean': total_ndcg / (index + 1)})
        pbar_val.update(1)
    pbar_val.close()



