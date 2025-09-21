import os
import json
import time
import torch
from loguru import logger
import random
import numpy as np

def fix_random_seed(seed):
    if seed is not None:
        random.seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark=False
        torch.backends.cudnn.deterministic=False

def get_subpath(dirpath,sort=False):
    path_list=os.listdir(dirpath)
    for i,path in enumerate(path_list):
        path_list[i]=os.path.normpath("%s/%s"%(dirpath,path))
    if sort:
        path_list.sort()
    return path_list
def join_path(first_path,second_path):
    path=os.path.normpath("%s/%s"%(first_path,second_path))
    return path

def dir_check(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def load_json(json_path,mode="r"):
    with open(json_path,mode) as f:
        cnnjson=json.load(f)
        f.close()
    return cnnjson

def read_txt(path):
    txt=open(path,encoding="utf-8")
    txt_list=[]
    for line in txt.readlines():
        line=line.strip("\n")
        line=line.split("\t")
        txt_list.append(line)
    return txt_list

def onehot_2_number(output):
    return torch.max(output,1)[1]
def number_2_onehot(number,nclass):
    batchsize=number.size[0]
    return torch.zeros(batchsize,nclass).scatter_(1,number,1)


def get_subfolder_names(folder_path):
    #输出文件夹下不加后缀的文件夹名称
    subfolders = []
    for item in os.listdir(folder_path):
        item_path = os.path.join(folder_path, item)
        if os.path.isdir(item_path) and '.' not in item:
            subfolders.append(item)
    return subfolders