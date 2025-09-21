import numpy as np
import random
import torch
import os
import pandas as pd


def normalize_l2(x, axis=1):
    '''x.shape = (num_samples, feat_dim)'''
    x_norm = np.linalg.norm(x, axis=axis, keepdims=True)
    x = x / (x_norm + 1e-8)
    return x

def fix_random_seed(seed):
    if seed is not None:
        random.seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark=False
        torch.backends.cudnn.deterministic=True

def get_subpath(dirpath,sort=False):
    path_list=os.listdir(dirpath)
    for i,path in enumerate(path_list):
        path_list[i]=os.path.normpath("%s/%s"%(dirpath,path))
    if sort:
        path_list.sort()
    return path_list
def get_subfolder_names(folder_path):
    #输出文件夹下不加后缀的文件夹名称
    subfolders = []
    for item in os.listdir(folder_path):
        item_path = os.path.join(folder_path, item)
        if os.path.isdir(item_path) and '.' not in item:
            subfolders.append(item)
    return subfolders


def save_dict_to_csv(dict_name,csv_file_path):
    max_length = max(len(lst) for lst in dict_name.values())
    for keys, values in dict_name.items():
        if len(values) >= max_length:
            continue
        else:
            for i in range(max_length - len(values)):
                values.append(0)
    df = pd.DataFrame(dict_name)
    df.to_csv(csv_file_path, index=False)