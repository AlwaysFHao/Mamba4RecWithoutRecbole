import os

import torch
from torch.utils.data import Dataset, DataLoader

import jsonlines


class AmazonDataset(Dataset):
    def __init__(self, root_path, max_len, split='train'):
        super(AmazonDataset, self).__init__()
        assert split in ['train', 'eval', 'test'], f"请输入正确的数据集类别，{split}不在['train', 'eval', 'test']中！"
        seq_dir = os.path.join(root_path, f'{split}_seq.jsonl')
        self.seq_list = []
        with open(seq_dir, 'r', encoding="utf8") as f:
            for item in jsonlines.Reader(f):
                self.seq_list.append(item)

        self.max_len = max_len
        self.user_num = 0
        self.item_num = 0
        with open(os.path.join(root_path, 'user2id.jsonl'), 'r', encoding="utf8") as f:
            for _ in jsonlines.Reader(f):
                self.user_num += 1
        with open(os.path.join(root_path, 'item2id.jsonl'), 'r', encoding="utf8") as f:
            for _ in jsonlines.Reader(f):
                self.item_num += 1

    def __len__(self):
        return len(self.seq_list)

    def __getitem__(self, idx):
        seq = self.seq_list[idx]
        user = seq[0]
        item_seq = [item[0] for item in seq[1]]
        time_stamps = [item[1] for item in seq[1]]

        # 生成padding_mask
        if len(item_seq) > self.max_len:
            padding_mask = torch.ones(self.max_len) == 0
            label = item_seq[self.max_len]
            time_stamps = time_stamps[:self.max_len]
            item_seq = item_seq[:self.max_len]
            length = self.max_len
        else:
            label = item_seq.pop()
            padding_mask = torch.cat((torch.ones(len(item_seq)), torch.zeros(self.max_len - len(item_seq)))) == 0
            length = len(item_seq)
            item_seq.extend([0 for _ in range(self.max_len - len(item_seq))])
            time_stamps.extend([0 for _ in range(self.max_len - len(time_stamps))])

        user = torch.tensor(user)
        item_seq = torch.tensor(item_seq)
        label = torch.tensor(label)
        length = torch.tensor(length)
        time_stamps = torch.tensor(time_stamps)

        return user, item_seq, label, length, padding_mask, time_stamps


if __name__ == '__main__':
    dataset = AmazonDataset(root_path='dataset/amazon/processed/Beauty', max_len=50, split='train')
    dataloader = DataLoader(dataset, batch_size=2, shuffle=False, drop_last=False)
    for i, data in enumerate(dataloader):
        print(data)
