import os
import argparse
import jsonlines
from tqdm import tqdm
from utils import get_sub_paths, load_inter_file, load_meta_file, filter_metas_by_inters, \
    filter_k_core_inters, group_inters_by_user

""" 2014亚马逊数据集处理，参考自 https://github.com/kz-song/MMSRec """


def parse_args():
    """
    定义基础参数
    :return: 基础参数
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw_path', default='../raw', type=str, help='raw data path')
    parser.add_argument('--processed_path', default='../processed', type=str, help='processed data path')
    # 最后保存的item信息中，text以及vision文件的前缀路径
    parser.add_argument('--prefix_path', default='./dataset/amazon/preprocess', type=str, help='prefix path')

    parser.add_argument('--k_core', default=5, type=int, help='filter inters by k core')

    parser.add_argument('--train_seq_outfile', default='train_seq.jsonl', type=str, help='processed train seq file')
    parser.add_argument('--eval_seq_outfile', default='eval_seq.jsonl', type=str, help='processed eval seq file')
    parser.add_argument('--test_seq_outfile', default='test_seq.jsonl', type=str, help='processed test seq file')
    parser.add_argument('--item2id_outfile', default='item2id.jsonl', type=str, help='processed item2id file')
    parser.add_argument('--user2id_outfile', default='user2id.jsonl', type=str, help='processed user2id file')
    args = parser.parse_args()
    return args


class AmazonProcessor(object):
    def __init__(self, args):
        """
        2014亚马逊数据集处理类
        :param args: 配置参数
        """
        self.args = args
        self.prefix_path = args.prefix_path

        self.raw_path = args.raw_path
        self.processed_path = args.processed_path

        self.target_train_seq_file = args.train_seq_outfile
        self.target_eval_seq_file = args.eval_seq_outfile
        self.target_test_seq_file = args.test_seq_outfile

        # 获取亚马逊数据集中的子类别文件夹
        self.sub_paths = get_sub_paths(self.raw_path)
        self.item2id = {}
        self.user2id = {}
        self.item2id_file = args.item2id_outfile
        self.user2id_file = args.user2id_outfile

    def write_seq_file(self, users, path):
        """
        生成并写入交互序列文件
        :param users: users的交互序列
        :param path: 子类别根目录
        :return:
        """
        print(f"Process Seq Data: {len(users)}")
        # 训练、验证和测试集
        train_seq_data = []
        eval_seq_data = []
        test_seq_data = []
        # 遍历users交互序列字典
        for id, interacts in tqdm(users.items()):
            uid = self.user2id[id]
            # 交互序列根据时间进行排序
            interacts = sorted(interacts, key=lambda item: item["time"])
            interacts = [(self.item2id[item["item"]], item["time"]) for item in interacts]

            # 生成当前用户的子序列作为训练集，最后两位空出
            for index in range(2, len(interacts) - 1):
                train_seq_data.append((uid, interacts[:index]))
            # 截取至倒数第二位作为验证集
            eval_seq_data.append((uid, interacts[:-1]))
            # 截取至最后一位作为测试集
            test_seq_data.append((uid, interacts[:]))
        # 生成子类别数据集路径
        target_path = os.path.join(self.processed_path, os.path.basename(os.path.normpath(path)))
        os.makedirs(target_path, exist_ok=True)

        # 保存训练集
        train_file = os.path.join(target_path, self.target_train_seq_file)
        with jsonlines.open(train_file, mode='w') as wfile:
            for line in train_seq_data:
                wfile.write(line)

        # 保存验证集
        eval_file = os.path.join(target_path, self.target_eval_seq_file)
        with jsonlines.open(eval_file, mode='w') as wfile:
            for line in eval_seq_data:
                wfile.write(line)

        # 保存测试集
        test_file = os.path.join(target_path, self.target_test_seq_file)
        with jsonlines.open(test_file, mode='w') as wfile:
            for line in test_seq_data:
                wfile.write(line)

    def _get_raw_file(self, path):
        """
        获取亚马逊子类别数据集的原始文件路径
        :param path: 子类别数据集的根目录
        :return: 原始文件路径（交互csv文件以及商品元数据）
        """
        raw_files = os.listdir(path)
        inter_file = [file for file in raw_files if file.endswith(".csv")][0]
        inter_file = os.path.join(path, inter_file)
        meta_file = [file for file in raw_files if file.startswith("meta") and file.endswith(".json.gz")][0]
        meta_file = os.path.join(path, meta_file)
        return inter_file, meta_file

    def _generate_item2id(self, meta, path):
        # 交互数据中存在的商品id，使用set去重（保证数据正确）
        items = set(meta)
        # set转list按照字符串进行排序
        items = sorted(items)
        for index, item in enumerate(items):
            self.item2id[item] = index + 1
        # 生成子类别数据集路径
        target_path = os.path.join(self.processed_path, os.path.basename(os.path.normpath(path)))
        os.makedirs(target_path, exist_ok=True)
        # 保存id映射表
        item2id_file = os.path.join(target_path, self.item2id_file)
        with jsonlines.open(item2id_file, mode='w') as wfile:
            for key in self.item2id:
                wfile.write((key, self.item2id[key]))

    def _generate_user2id(self, users_inter, path):
        # 交互数据中存在的用户id
        users = set()
        # 统计所有的商品id
        for user_id in tqdm(users_inter.keys(), desc="generate user2id file"):
            users.add(user_id)
        # set转list按照字符串进行排序
        users = sorted(users)
        for index, user in enumerate(users):
            self.user2id[user] = index
        # 生成子类别数据集路径
        target_path = os.path.join(self.processed_path, os.path.basename(os.path.normpath(path)))
        os.makedirs(target_path, exist_ok=True)
        # 保存id映射表
        user2id_file = os.path.join(target_path, self.user2id_file)
        with jsonlines.open(user2id_file, mode='w') as wfile:
            for key in self.user2id:
                wfile.write((key, self.user2id[key]))

    def process(self):
        # 遍历所有已经存在的子类别目录
        for path in self.sub_paths:
            print(f"\n-----Processing data {path}")
            # 获取原始文件路径
            inter_file, meta_file = self._get_raw_file(path)
            # 根据交互csv文件得到交互数据集合 set(set(user, item, rate, time))
            inters = load_inter_file(inter_file)
            # 加载meta文件得到商品列表
            metas = load_meta_file(meta_file)
            # k-core过滤
            inters = filter_k_core_inters(inters, self.args.k_core, self.args.k_core)
            # 根据现有的交互数据过滤商品数据
            metas = filter_metas_by_inters(metas, inters)

            # 生成item2id的映射
            self._generate_item2id(metas, path)
            # 生成users交互序列
            users = group_inters_by_user(inters)
            # 生成user2id的映射
            self._generate_user2id(users, path)
            # 切分生成交互序列数据集
            self.write_seq_file(users, path)


if __name__ == '__main__':
    # 初始化配置参数
    args = parse_args()
    # 实例化亚马逊处理对象
    api = AmazonProcessor(args)
    # 调用处理方法
    api.process()
