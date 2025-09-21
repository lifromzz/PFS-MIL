import torch
import os

import tqdm
from torch_geometric.data import Dataset, Data, InMemoryDataset
from typing import Union, List, Tuple
import pandas as pd


class GraphDataset(Dataset):
    def __init__(self,
                 root,
                 raw,
                 processed_stage,
                 split_path=None,
                 sheet_name="train",
                 transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root,transform, pre_transform, pre_filter)

        # 从csv文件中提取某一个sheet表中的file_name列和label列数据
        self.root=root
        self.raw=raw
        self.processed_stage=processed_stage
        self.split_path=split_path
        self.sheet_name=sheet_name


        print("load datset: {}".format(self.processed_stage))
        self.pre_load()

    def pre_load(self):
        file_labels_df = pd.read_excel(self.split_path, sheet_name=self.sheet_name)
        train_slide_names = file_labels_df['file_name'].tolist()

        data = os.path.join(self.root, self.processed_stage)
        pt_files = [data + "/" + slide_name + ".pt" for slide_name in train_slide_names]
        self.data = []
        for pt_path in tqdm.tqdm(pt_files):
            self.data.append(torch.load(pt_path))

    @property
    def has_process(self) -> bool:
        return False

    @property
    def raw_file_names(self) -> Union[str, List[str], Tuple]:
        # return os.listdir(os.path.join(self.root, "raw"))
        return os.listdir(os.path.join(self.root, "raw"))

    @property
    def processed_dir(self) -> str:

        path=os.path.join(self.root, self.processed_stage)
        return path

    @property
    def processed_file_names(self) -> Union[str, List[str], Tuple]:
        file_labels_df = pd.read_excel(self.split_path, sheet_name=self.sheet_name)
        train_slide_names = file_labels_df['file_name'].tolist()

        data = os.path.join(self.root, self.processed_stage)
        pt_files = [data + "/" + slide_name + ".pt" for slide_name in train_slide_names]
        return pt_files

    def len(self) -> int:
        return len(self.processed_file_names)  #所改的第三个地方

    # def get(self, idx: int) -> Data:
    #     return torch.load(self.processed_paths[idx])
    def get(self, idx: int) -> Data:
        return self.data[idx]