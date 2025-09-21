import os

import torch
import glob
import random
import numpy as np
import pandas as pd
from tqdm import tqdm

from torch.utils.data import Dataset


class base(Dataset):
    def __init__(self, data, split_path, sheet_name="train"):
        super().__init__()

        # 从csv文件中提取某一个sheet表中的file_name列和label列数据
        file_labels_df = pd.read_excel(split_path, sheet_name=sheet_name)
        train_slide_names = file_labels_df['file_name'].tolist()
        pt_files = [data + "/" + slide_name + ".pt" for slide_name in train_slide_names]
        self.pt_files = pt_files
        self.pt_label = file_labels_df['label'].tolist()
        print("finishing")

    def __len__(self):
        return len(self.pt_label)


class myDatasetInMemory(base):
    def __init__(self, data, split_path, sheet_name="train", batch_size=16):
        super().__init__(data, split_path, sheet_name)

        self.batch_size = batch_size

        self.data = []
        self.label = []
        self.batch = []

        bs_data = []
        bs_label = []
        bs_batch = []

        numbers = list(range(len(self.pt_files)))  # 生成0到10的整数列表
        random.shuffle(numbers)  # 打乱列表

        print("loading {} data".format(sheet_name))
        for c, i in enumerate(tqdm(numbers)):
            l34 = torch.load(self.pt_files[i], map_location="cpu")
            l34_batch = torch.ones(size=(len(l34),), dtype=torch.int64) * (c % self.batch_size)
            bs_label.append(self.pt_label[i])
            bs_batch.append(l34_batch)
            bs_data.append(l34)

            if (c + 1) % batch_size == 0 and (c + 1) < len(self.pt_files):

                self.data.append(bs_data)
                self.label.append(bs_label)
                self.batch.append(bs_batch)
                bs_data = []
                bs_label = []
                bs_batch = []
            elif (c + 1) == len(self.pt_files):
                self.data.append(bs_data)
                self.label.append(bs_label)
                self.batch.append(bs_batch)
        print("finishing")

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        return self.data[index], self.label[index], self.batch[index]


class myDataset(base):
    def __init__(self, data, split_path, sheet_name="train"):
        super().__init__(data, split_path, sheet_name)
        pass

    def __getitem__(self, index):
        path = self.pt_files[index]
        data = torch.load(path, map_location="cpu")
        return data, self.pt_label[index]


if __name__ == "__main__":
    repre_dir = "D:\\HNSZL Train\\represention\\r50_level0_224\\stage3"
    test_split_xlsl = "D:\\project\\laten_mamba_main\\data\\LUNG\\luad_lusc_normal_Lung_train_file_labels.xlsx"
    test_sheet = "train"

    dataset = myDataset(data=repre_dir,
                        split_path=test_split_xlsl,
                        sheet_name=test_sheet)
    for i in range(len(dataset)):
        print(dataset.__getitem__(i))
    # dataset = EmbededFeatsDataset('/your/path/to/CAMELYON16/', mode='test')
