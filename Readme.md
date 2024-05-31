# Mamba4Rec复现-无需Recbole环境

作者: 范昊

author: Hao Fan

## 简介
本仓库在非Recbole环境下复现了[Mamba4Rec](https://arxiv.org/abs/2403.03900)模型的性能，并提供了Amazon 2014数据集的预处理方法。
- `模型`: 本仓库从MambaBlock开始一步步重构了Mamba4Rec模型结构（作者能力所限，无法实现selective_scan_cuda，涉及cuda编程），同时给出了详细注释，并参考Recbole序列推荐模型，实现了模型训练与预测逻辑；
- `数据集`: 本仓库针对Amazon 2014评论数据集实现了序列推荐的数据预处理流程。
---
## 引用
- Mamba4Rec模型结构实现参考自Mamba4Rec官方源码： [https://github.com/chengkai-liu/Mamba4Rec](https://github.com/chengkai-liu/Mamba4Rec)；
- 重构MambaBlock参考自:
  - Mamba源码：[https://github.com/state-spaces/mamba](https://github.com/state-spaces/mamba)；
  - Mamba-Py源码：[https://github.com/alxndrTL/mamba.py](https://github.com/alxndrTL/mamba.py)；
- 序列推荐模型训练以及预测逻辑实现参考自Recbole源码：[https://github.com/RUCAIBox/RecBole](https://github.com/RUCAIBox/RecBole)；
- Amazon 2014评论数据集预处理实现参考自MMSRec源码：[https://github.com/kz-song/MMSRec](https://github.com/kz-song/MMSRec)；
- Hit和NDCG评价指标实现参考自TiCoSeRec源码：[https://github.com/KingGugu/TiCoSeRec](https://github.com/KingGugu/TiCoSeRec)。

Mamba4Rec论文bibtex引用如下：
```
@article{liu2024mamba4rec,
      title={Mamba4Rec: Towards Efficient Sequential Recommendation with Selective State Space Models}, 
      author={Chengkai Liu and Jianghao Lin and Jianling Wang and Hanzhou Liu and James Caverlee},
      journal={arXiv preprint arXiv:2403.03900},
      year={2024}
}
```
---
## 基础环境要求
- Linux（本案例使用Ubuntu-22.0.4发行版本）
  - 请先确定系统的GLIBC版本大于等于2.32（本案例使用2.35）， 否则会导致python无法正常import动态链接库（python >= 3.7 import动态链接库需要 GLIBC >= 2.32），
  如需查看GLIBC版本可使用以下命令查看：
  ```shell
  ldd --version
  ```
- Python >= 3.9（or 3.8?） （本案例使用python-3.10）
- CUDA >= 11.6 （本案例使用CUDA-11.8）
- Pytorch >= 1.12.1 （本案例使用torch-2.3.0）
- jsonlines == 2.0.0
- Mamba （如遇安装问题，可参考：[https://github.com/AlwaysFHao/Mamba-Install](https://github.com/AlwaysFHao/Mamba-Install) ）
  - causal_conv1d >= 1.2.0 （本案例使用causal_conv1d-1.2.0）
  - mamba_ssm >= 1.2.0 （本案例使用mamba_ssm-1.2.0）
---
## 数据集准备
努力编写ing，请等待更新...

---
## 项目结构介绍
努力编写ing，请等待更新...

---
## Usage
努力编写ing，请等待更新...

