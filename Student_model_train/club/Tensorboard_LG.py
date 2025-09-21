from torch.utils.tensorboard import SummaryWriter
from Student_model_train.club.utils import get_subpath,get_subfolder_names
import os

class tensorboard_lg():
    def __init__(self,tensorboard_folder):
        self.tensorboard_folder=tensorboard_folder
        self.recoder_dict={}
        self.mini_batch_count = 0

        if not os.path.exists(tensorboard_folder):
            os.makedirs(tensorboard_folder)
    def init_tensorbard(self,seed):
        next_tensorboard_log = os.path.join(self.tensorboard_folder, str(seed))
        if not os.path.exists(next_tensorboard_log):
            os.makedirs(next_tensorboard_log)
        # 创建写入器
        self.writer = SummaryWriter(next_tensorboard_log)
    def next_tensorbard(self):
        subfolder_names_list=get_subfolder_names(self.tensorboard_folder)
        if len(subfolder_names_list)==0:
            max_index=0
        else:
            max_index=max([int(i) for i in subfolder_names_list])
        next_tensorboard_log=os.path.join(self.tensorboard_folder,str(max_index+1))
        if not os.path.exists(next_tensorboard_log):
            os.makedirs(next_tensorboard_log)
        #创建写入器
        self.writer = SummaryWriter(next_tensorboard_log)

    def recoder_result(self,result_dict):
        for k, v in result_dict.items():
            if "prob" in k:
                continue
            # 判断键是否在字典中，如果不在则添加，如果在则更新值
            if k in self.recoder_dict:
                self.recoder_dict[k] += v
            else:
                self.recoder_dict[k] = v

    def refresh_log(self, epoch):
        for k in self.recoder_dict.keys():
            if ("acc" in k) or ("loss" in k):
                v=self.recoder_dict[k]/self.mini_batch_count
                self.recoder_dict[k]=v
                self.writer.add_scalar(k, v, epoch)