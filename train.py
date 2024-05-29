# -*- coding: utf-8 -*-            
# @Author : Hao Fan
# @Time : 2024/5/29
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

    device = (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))

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

    train_dataset = AmazonDataset(root_path=root_path, max_len=max_len, split='train')
    user_num = train_dataset.user_num
    item_num = train_dataset.item_num
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
    train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False)
    val_dataset = AmazonDataset(root_path=root_path, max_len=max_len, split='eval')
    val_dataloader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)

    # 计算总共批次数(不足batch_size的直接舍弃)
    epoch_step = len(train_dataset) // batch_size
    val_epoch_step = len(val_dataset) // batch_size
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_hit, best_ndcg = max_val_hit, -1.0

    for epoch in range(Epoch):
        logger.info("开始训练")
        model.train()
        total_loss = 0.0
        # 实例化进度条
        pbar = tqdm(total=epoch_step, desc=f'Epoch {epoch + 1}/{Epoch}',
                    postfix={'loss': '?', "loss_mean": '?'}, mininterval=0.3)

        for index, (_, user_item_seq, label, length, _, _) in enumerate(train_dataloader):
            # 舍弃不足batch_size的批次
            if index >= epoch_step:
                break

            user_item_seq = user_item_seq.to(device)
            train_label = label.to(device)
            length = length.to(device)

            optimizer.zero_grad()
            # 模型正向传播
            loss = model.calculate_loss(
                item_seq=user_item_seq,
                item_seq_len=length,
                labels=train_label
            )
            loss.backward()
            optimizer.step()
            # 计算平均损失
            total_loss += loss.item()
            mean_loss = total_loss / (index + 1)
            # 更新进度条
            pbar.set_postfix(**{'loss': loss.item(),
                                'loss_mean': mean_loss})
            pbar.update(1)
        pbar.close()

        logger.info("开始验证")
        model.eval()
        pbar_val = tqdm(total=val_epoch_step, desc=f'Epoch {epoch + 1}/{Epoch}',
                        postfix={'hit': '?', 'ndcg': '?', 'hit_mean': '?', 'ndcg_mean': '?'},
                        mininterval=0.3)
        total_hit, total_ndcg = 0.0, 0.0

        for index, (_, seq, label, length, _, _) in enumerate(val_dataloader):
            # 舍弃不足batch_size的批次
            if index >= val_epoch_step:
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

        if total_hit / val_epoch_step > best_hit:
            best_hit = total_hit / val_epoch_step
        if total_ndcg / val_epoch_step > best_ndcg:
            best_ndcg = total_ndcg / val_epoch_step

        if epoch % 10 == 0:
            logger.info(f'当前最好的hit:{best_hit}, 最好的ndcg:{best_ndcg}')
        if total_hit / val_epoch_step > max_val_hit:
            logger.info(f'将最好的模型保存到{model_saved_path}')
            max_val_hit = total_hit / val_epoch_step
            model_dict = model.state_dict()
            model_dict['max_val_hit'] = total_hit / val_epoch_step
            torch.save(model_dict, os.path.join(model_saved_path, model_saved_name))

        torch.save(model.state_dict(), os.path.join(model_saved_path, f"{logger_name}_last_epoch_model.pth"))
