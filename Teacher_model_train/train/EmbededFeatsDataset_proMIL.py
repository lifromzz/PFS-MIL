import os

import torch
import glob
import random
import numpy as np
import pandas as pd
from tqdm import tqdm

from torch.utils.data import Dataset


class EmbededFeatsDataset(Dataset):
    def __init__(self,data, split_path, sheet_name="train",batch=False):
        super().__init__()

        # 从csv文件中提取某一个sheet表中的file_name列和label列数据
        file_labels_df = pd.read_excel(split_path, sheet_name=sheet_name)
        train_slide_names = file_labels_df['file_name'].tolist()
        pt_files = [data + "/" + slide_name + ".pt" for slide_name in train_slide_names]
        self.pt_files=pt_files
        self.batch=batch

        # self.data=[torch.load(path,map_location="cpu") for path in tqdm(pt_files,desc="loading {} data".format(sheet_name))]
        #
        self.data=[]
        for path in tqdm(pt_files, desc="loading {} data".format(sheet_name)):
            # print(path)
            self.data.append(torch.load(path,map_location="cpu"))
        self.label = file_labels_df['label'].tolist()
        print("finishing")

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        data=self.data[index]
        if self.batch:
            data = data["l34"]
            data=torch.squeeze(data)
            batch=torch.ones(size=(len(data),),dtype=torch.int64)*index
            return data,self.label[index],batch
        return data, self.label[index]



class EmbededFeatsDataset_ProMIL(Dataset):
    def __init__(self,data, split_path, sheet_name="train",batch_size=16,stage="l34"):
        super().__init__()

        # 从csv文件中提取某一个sheet表中的file_name列和label列数据
        file_labels_df = pd.read_excel(split_path, sheet_name=sheet_name)
        train_slide_names = file_labels_df['file_name'].tolist()
        pt_files = [data + "/" + slide_name + ".pt" for slide_name in train_slide_names]
        pt_label = file_labels_df['label'].tolist()

        self.batch_size=batch_size


        self.data=[]
        self.label=[]
        self.batch=[]

        bs_data = []
        bs_label=[]
        bs_batch=[]

        numbers = list(range(len(pt_files)))  # 生成0到10的整数列表
        random.shuffle(numbers)  # 打乱列表

        print("loading {} data".format(sheet_name))
        for c,i in enumerate(tqdm(numbers)):
            if stage:
                l34 = torch.load(pt_files[i], map_location="cpu")
                if "l34" in l34.keys() and stage not in l34.keys():
                    if stage=="l3":
                        l34=l34["l34"][:,:1024]
                    elif stage=="l4":
                        l34=l34["l34"][:,1024:]
                else:
                     l34=l34[stage]
            else:
                l34 = torch.load(pt_files[i], map_location="cpu")

            l34_batch = torch.ones(size=(len(l34),), dtype=torch.int64) * (c%self.batch_size)
            bs_label.append(pt_label[i])
            bs_batch.extend(l34_batch)
            bs_data.extend(l34)

            if (c+1) % batch_size==0 and (c+1)<len(pt_files):
                if len(bs_label)==0:
                    break
                self.data.append(torch.stack(bs_data))
                self.label.append(torch.tensor(bs_label))
                self.batch.append(torch.tensor(bs_batch))
                bs_data=[]
                bs_label=[]
                bs_batch=[]
            elif (c+1)==len(pt_files):
                if len(bs_label)==0:
                    break
                self.data.append(torch.stack(bs_data))
                self.label.append(torch.tensor(bs_label))
                self.batch.append(torch.tensor(bs_batch))
        print("finishing")

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):

        return self.data[index], self.label[index], self.batch[index]



class EmbededFeatsDataset2(Dataset):
    def __init__(self,data, split_path, sheet_name="train"):
        super().__init__()

        # 从csv文件中提取某一个sheet表中的file_name列和label列数据
        file_labels_df = pd.read_excel(split_path, sheet_name=sheet_name)
        train_slide_names = file_labels_df['file_name'].tolist()
        self.pt_files = [data + "/" + slide_name + ".pt" for slide_name in train_slide_names]
        self.label = file_labels_df['label'].tolist()
        print("finishing")

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        path=self.pt_files[index]
        data=torch.load(path,map_location="cpu")
        return data,self.label[index]
        # pt_name=os.path.basename(path)
        # return pt_name,self.label[index]


if __name__ == "__main__":
    repre_dir = "D:\\HNSZL Train\\represention\\r50_level0_224\\stage3"
    test_split_xlsl = "D:\\project\\laten_mamba_main\\data\\LUNG\\luad_lusc_normal_Lung_train_file_labels.xlsx"
    test_sheet = "train"

    dataset = EmbededFeatsDataset2(data=repre_dir,
                                   split_path=test_split_xlsl,
                                   sheet_name=test_sheet)
    for i in range(len(dataset)):
        print(dataset.__getitem__(i))
    # dataset = EmbededFeatsDataset('/your/path/to/CAMELYON16/', mode='test')