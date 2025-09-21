import pandas as pd
import csv
import os

import tqdm


def init_csv(label_dict,path_csv="/test_csv.csv",save=False):
    if save==False:
        return
    based_list = ["AUC", "ACC", "F1Score", "Recall", "Precision", "Specificity"]
    metirc_list = based_list + ["CohenKappa"]
    for b in based_list:
        for c in label_dict:
            metirc_list.append(c + "-" + b)
    input_str = "{}," * (len(metirc_list) - 1) + "{}\n"
    write_csv(path_csv, input_str.format(*metirc_list), "w")

def write_csv(filepath,head,mode):
    """
    将head以mode写入filepath中
    :param filepath: csv文件的路径,可以为绝对路径也可以是相对路径
    :param head: 写入csv文件的内容,可以是
    :param mode:w:原文件存在则覆盖源文件,从头开始写.a:从文件末尾开始追加
    :return:
    """
    fconv=open(filepath,mode)
    fconv.write(head)
    fconv.close()
def read_csv(csv_path):
    """
    读取csv文件,读取方式如下,
    csv_dict=base_func.read_csv("./result/weak_simsiam.csv")
    print(csv_dict.keys())
    :param csv_path:
    :return:返回一个字典，csv文件中一列构成字典中的一个键值对，列名为键，整个一列为值
    """
    #df=pd.read_csv(csv_path,index_col=0)
    df = pd.read_csv(csv_path)
    column_names=[name for name in df.columns] #提取csv文件中的列名
    csv_dict={}
    for name in column_names: #
        index=df[name].notnull() #找到name列中所有不为空的行索引
        csv_dict[name]=df[name][index].values.tolist()
    return csv_dict

def read_csv_row(csv_path):
    with open(csv_path,"r") as csvfile: #encoding=“utf-8”
        reader = csv.reader(csvfile)
        rows = [row for row in reader]
    return rows
def copy_csv(source_csv_path,dest_csv_path):
    if os.path.exists(dest_csv_path):
        return
    source_csv=read_csv_row(source_csv_path)
    pbar = tqdm.tqdm(source_csv)
    for id,label in pbar:
        pbar.set_description("copy csv")
        write_csv(dest_csv_path,"{},{}\n".format(id,label),"a")