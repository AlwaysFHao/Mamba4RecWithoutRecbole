# -*- coding: utf-8 -*-            
# @Author : Hao Fan
# @Time : 2024/5/29
import logging

import math

import torch

"""
The implementation of the evaluation index part refers to TiCoSeRec: https://github.com/KingGugu/TiCoSeRec
"""

def setting_logging(log_name):
    """
    设置日志
    :param log_name: 日志名
    :return: 可用日志
    """
    # 第一步：创建日志器对象，默认等级为warning
    logger = logging.getLogger(log_name)
    logging.basicConfig(level="INFO")

    # 第二步：创建控制台日志处理器
    console_handler = logging.StreamHandler()

    # 第三步：设置控制台日志的输出级别,需要日志器也设置日志级别为info；----根据两个地方的等级进行对比，取日志器的级别
    console_handler.setLevel(level="WARNING")

    # 第四步：设置控制台日志的输出格式
    console_fmt = "%(name)s--->%(asctime)s--->%(message)s--->%(lineno)d"
    fmt1 = logging.Formatter(fmt=console_fmt)
    console_handler.setFormatter(fmt=fmt1)

    # 第五步：将控制台日志器，添加进日志器对象中
    logger.addHandler(console_handler)

    return logger


def recall_at_k(actual, predicted, topk):
    sum_recall = 0.0
    _, predicted = torch.topk(predicted, k=topk, dim=1)
    predicted = predicted.cpu().numpy()
    num_users = len(predicted)
    true_users = 0
    for i in range(num_users):
        act_set = set(actual[i])
        pred_set = set(predicted[i])
        if len(act_set) != 0:
            sum_recall += len(act_set & pred_set) / float(len(act_set))
            true_users += 1
    return sum_recall / true_users


def idcg_k(k):
    res = sum([1.0 / math.log(i + 2, 2) for i in range(k)])
    if not res:
        return 1.0
    else:
        return res


def ndcg_k(actual, predicted, topk):
    res = 0
    _, predicted = torch.topk(predicted, k=topk, dim=1)
    predicted = predicted.cpu().numpy()
    for user_id in range(len(actual)):
        k = min(topk, len(actual[user_id]))
        idcg = idcg_k(k)
        dcg_k = sum([int(predicted[user_id][j] in
                         set(actual[user_id])) / math.log(j + 2, 2) for j in range(topk)])
        res += dcg_k / idcg
    return res / float(len(actual))


def get_full_sort_score(answers, pred_list, topk):
    recall = recall_at_k(answers, pred_list, topk)
    ndcg = ndcg_k(answers, pred_list, topk)
    return recall, ndcg

