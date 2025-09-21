import torch
import random
from random import shuffle

from tqdm import tqdm

from Student_model_train.club.checkpoint_LG import checkpoint_lg
from Student_model_train.dataset.EmbededFeatsDataset import base
class stage2_dataset(base):
    def __init__(self,
                 data,
                 split_path,
                 sheet_name,
                 model,
                 ckg_path,
                 batch_size,
                 top_per,
                 bottom_per,
                 need_shuffle,
                 stage="l3"
                 ):
        super().__init__(data, split_path, sheet_name)

        self.ckp = checkpoint_lg()
        self.data=data
        self.sheet_name=sheet_name
        self.batch_size=batch_size
        self.top_per=top_per
        self.bottom_per=bottom_per
        self.train_step =( len(self.pt_files) // self.batch_size)+1
        self.need_shuffle=need_shuffle
        self.stage=stage

        self.model=self.ckp.load_checkpoint(model=model,path=ckg_path)
        self.build_data()

    def split_feature_by_attn_sing_bag(self,bag,attn):
        """

        :param bag:
        :param attn:
        :param istrain:
        :return: 前10%的实例和10~90%的实例融合后的包特征
        """
        num_top_nodes = int(len(bag) * self.top_per)  # 前20%节点数量
        num_bottom_nodes = int(len(bag) * self.bottom_per)  # 后20%节点数量
        attn=torch.squeeze(attn,1)
        # 找出attn_x占前20%的节点索引
        attn_clone=attn.clone().detach()
        _, top_indices = torch.topk(attn_clone, num_top_nodes)
        _, bottom_indices = torch.topk(attn_clone, num_bottom_nodes, largest=False)
        mid_indices = list(set(range(len(attn))) - set(top_indices.tolist() + bottom_indices.tolist()))
        mid_indices = torch.tensor(mid_indices)

        top_instance = bag[top_indices]
        mid_instance = bag[mid_indices]

        return top_instance,mid_instance
    def random_shuffle(self,features,labels):
        # 将特征和标签打包成元组的列表
        data = list(zip(features, labels))
        # 随机打乱
        random.shuffle(data)
        # 解压缩得到打乱后的特征和标签
        shuffled_features, shuffled_labels = zip(*data)
        return torch.stack(shuffled_features), torch.stack(shuffled_labels)


    def build_data(self):
        self.Hight_value_groups_x=[]
        self.Hight_value_groups_y=[]
        self.Compex_value_groups_x=[]
        self.Compex_value_groups_y=[]

        numbers = list(range(len(self.pt_files)))  # 生成0到10的整数列表
        if self.need_shuffle:
            random.shuffle(numbers)  # 打乱列表

        print("building data {}".format(self.sheet_name))
        for c, i in enumerate(tqdm(numbers)):
            instances = torch.load(self.pt_files[i], map_location="cpu")
            if isinstance(instances,dict):
                if "stage34" in self.data:
                    instances=instances["l34"]
                elif "stage4" in self.data:
                    instances=instances["l4"]
                elif "stage3" in self.data:
                    instances=instances["l3"]
                else:
                    instances=instances[self.stage]


            instances, instances_attns = self.model(instances,None,False)
            top_instance,mid_instance=self.split_feature_by_attn_sing_bag(instances,instances_attns)
            self.Hight_value_groups_x.append(top_instance)
            self.Hight_value_groups_y.append([self.pt_label[i]]*len(top_instance))

            self.Compex_value_groups_x.append(mid_instance)
            self.Compex_value_groups_y.append(self.pt_label[i])

        print("finishing")
    def get_random_indexs(self):
        indexs = list(range(len(self.pt_files)))
        if self.need_shuffle:
            shuffle(indexs)
        new_random_index=[]
        start=0
        for i in range(self.train_step):
            new_random_index.append(indexs[start:start+self.batch_size])

            start+=self.batch_size
        return new_random_index

    def get_batch(self,index):
        if len(index)==1:
            Hight_value_groups_x=self.Hight_value_groups_x[index[0]]
        elif len(index)>1:
            Hight_value_groups_x=torch.cat([self.Hight_value_groups_x[i] for i in index])

        Hight_value_groups_y=[]
        for i in index:
            Hight_value_groups_y.extend(self.Hight_value_groups_y[i])
        Hight_value_groups_y=torch.tensor(Hight_value_groups_y)
        if len(Hight_value_groups_x)!=0:
            Hight_value_groups_x,Hight_value_groups_y=self.random_shuffle(Hight_value_groups_x,Hight_value_groups_y)
        if len(index)==1:
            Compex_value_groups_x=self.Compex_value_groups_x[index[0]]
        elif len(index)>1:
            Compex_value_groups_x=torch.cat([self.Compex_value_groups_x[i] for i in index])
        Compex_value_groups_y=torch.tensor([self.Compex_value_groups_y[i] for i in index])
        batch=[]
        for c,i in enumerate(index):
            group=self.Compex_value_groups_x[i]
            bag_batch = torch.ones(size=(len(group),), dtype=torch.int64) * (c % self.batch_size)
            batch.append(bag_batch)
        batch=torch.cat(batch)
        return Hight_value_groups_x,Hight_value_groups_y,Compex_value_groups_x,Compex_value_groups_y,batch
