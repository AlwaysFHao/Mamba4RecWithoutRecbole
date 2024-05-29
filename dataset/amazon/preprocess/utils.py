import gzip
import os

from tqdm import tqdm

"""2014亚马逊数据集处理，参考自 https://github.com/kz-song/MMSRec """


def get_sub_paths(path):
    """
    获取亚马逊数据集中的子类别文件夹
    :param path: 亚马逊数据集的原始文件根路径
    :return: 所有子类别数据集的路径
    """
    sub_paths = [os.path.join(path, sub_path) for sub_path in os.listdir(path)]
    results = [sub_path for sub_path in sub_paths if os.path.isdir(sub_path)]
    return results


def _parse_csv_line(line: str):
    """
    按行读取csv文件
    :param line: 当前csv行数据字符串
    :return: 切分后的交互信息字典
    """
    data = line.strip().split(",")
    return {"user": str(data[0]), "item": str(data[1]), "rate": float(data[2]), "time": int(data[3])}


def load_inter_file(inter_file):
    """
    根据交互csv文件得到交互数据
    :param inter_file: 交互csv文件路径
    :return:
    """
    # 断言，必须是csv文件
    assert inter_file.endswith(".csv"), '当前交互数据非csv文件，请下载csv版本的交互数据'

    # 交互数据集合
    inters = set()
    with open(inter_file, "r", encoding="utf-8") as fobj:
        # csv文件的readlines()方法返回行数据列表
        for line in tqdm(fobj.readlines(), desc=f"load inter file {inter_file}"):
            # 将行数据字符串解析成字典数据
            data = _parse_csv_line(line)
            user = data["user"]
            item = data["item"]
            rate = data["rate"]
            time = data["time"]
            # 交互字典转为元组添加进交互集合
            inters.add((user, item, rate, time))
    print('Total inters:', len(inters))
    return inters


def _parse_gz_line(line):
    """
    按行解析json行数据
    :param line: json字符串
    :return: 字典
    """
    # eval方法将传入的字符串直接转换成python代码执行，这里传入的是json字符串，所以直接返回字典
    return eval(line)


def load_meta_file(meta_file):
    """
    加载商品元数据
    :param meta_file: 商品元数据文件
    :return: 商品数据列表
    """
    assert meta_file.endswith(".json.gz"), '请传入正确的商品元数据文件，格式为.gz'

    # 商品数据
    metas = []
    # 解压gzip文件
    gzip_file = gzip.open(meta_file, 'r')
    # gzip_file为可迭代对象，每次迭代返回行数据
    for line in tqdm(gzip_file, desc=f"load meta file {meta_file}"):
        # 按行解析行数据，这里的line实际上就是一行json字典
        data = _parse_gz_line(line)
        # 把商品元数据中的商品id取出
        item = str(data["asin"])
        # 加入商品数据
        metas.append(item)
    return metas


def filter_metas_by_inters(metas, inters):
    """
    根据现有的交互数据过滤商品数据
    :param metas: 商品列表
    :param inters: 交互列表
    :return: 过滤后的商品元数据
    """
    # 交互数据中存在的商品id
    items = set()
    # 统计所有的商品id
    for inter in tqdm(inters, desc="filter metas by inters"):
        items.add(inter[1])
    # 新的商品数据
    new_metas = []
    # 遍历旧的商品数据
    for id in metas:
        # 判断当前商品是否在交互数据中，存在则加入新的商品数据
        if id in items:
            new_metas.append(id)
    return new_metas


def filter_k_core_inters(inters: set, user_inter_threshold=5, item_inter_threshold=5):
    """
    k-core过滤
    :param inters: 交互集合
    :param user_inter_threshold: 用户最少交互数
    :param item_inter_threshold: 商品最少交互数
    :return:
    """
    print(f"Filter K core: user {user_inter_threshold}, item {item_inter_threshold}")
    while True:
        # 新建用于统计user和item交互次数的字典
        user_count = {}
        item_count = {}
        # 遍历交互数据
        for inter in inters:
            # 统计累加用户交互数
            if inter[0] not in user_count:
                user_count[inter[0]] = 1
            else:
                user_count[inter[0]] += 1
            # 统计累加商品交互数
            if inter[1] not in item_count:
                item_count[inter[1]] = 1
            else:
                item_count[inter[1]] += 1
        # 新交互列表
        new_inters = []
        for inter in inters:
            # 判断是否满足最少交互数
            if user_count[inter[0]] >= user_inter_threshold and \
                    item_count[inter[1]] >= item_inter_threshold:
                # 如果都满足则添加进列表
                new_inters.append(inter)
        print(f"\tFilter: {len(inters)} inters to {len(new_inters)} inters")
        # 判断如果新的交互数据和旧的交互数据数量一致，则返回过滤后的数据
        if len(new_inters) == len(inters):
            return new_inters
        # 新的交互数据和旧的交互数据数量不一致，则替换旧数据再进行一轮筛选
        inters = new_inters


def group_inters_by_user(inters):
    """
    生成users的交互序列
    :param inters: 交互数据
    :return: users的交互序列
    """
    # users交互序列字典
    users = {}
    for inter in tqdm(inters, desc="group inters by user"):
        # 如果不在users中，则初始化一个空列表，相当于初始化
        if inter[0] not in users:
            users[inter[0]] = []
        # 添加进users交互序列中
        users[inter[0]].append({"item": inter[1], "time": inter[3]})
    return users

