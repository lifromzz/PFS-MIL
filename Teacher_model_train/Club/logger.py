import os
import csv
import datetime
from torchsummary import summary
from utils import util
from torch.utils.tensorboard import SummaryWriter

class logger:
    def __init__(self,dataset_name,log_root="log"):
        self.log_root=log_root
        if not os.path.exists(self.log_root):
            os.makedirs(self.log_root)

        self.init_data_folder(dataset_name)


    def init_data_folder(self,dataset_name):
        # 获取当前日期和时间
        # current_datetime = datetime.datetime.now()
        # # 格式化输出
        # formatted_datetime = current_datetime.strftime("%Y-%m-%d-%H-%M")
        # current_log=formatted_datetime+"-"+model_name

        current_log=os.path.join(self.log_root,dataset_name)
        if not os.path.exists(current_log):
            os.makedirs(current_log)
        self.current_log=current_log

        #创建模型权重文件夹
        checkpoint_folder=os.path.join(current_log,"checkpoint")
        if not os.path.exists(checkpoint_folder):
            os.makedirs(checkpoint_folder)
        self.checkpoint_folder=checkpoint_folder

        #创建训练结果文件夹
        result_folder=os.path.join(current_log,"results")
        if not os.path.exists(result_folder):
            os.makedirs(result_folder)

        tensorboard_folder=os.path.join(current_log,"tensorboard")
        if not os.path.exists(tensorboard_folder):
            os.makedirs(tensorboard_folder)
        self.tensorboard_folder=tensorboard_folder


    def summary_model(self,model):
        # 打印模型结构和参数量
        summary_str = str(summary(model, (3, 224, 224)))  # 输入的维度根据你的模型输入维度来设定
        print(summary_str)

        model_summary_path=os.path.join(self.current_log,"model_summary.txt")
        # 将结果保存在txt文件中
        with open(model_summary_path, "w", encoding="utf-8") as f:
            f.write(summary_str)

    def next_tensorbard(self):
        subfolder_names_list=util.get_subfolder_names(self.tensorboard_folder)
        if len(subfolder_names_list)==0:
            max_index=0
        else:
            max_index=max([int(i) for i in subfolder_names_list])
        next_tensorboard_log=os.path.join(self.tensorboard_folder,str(max_index+1))
        self.writer = SummaryWriter(next_tensorboard_log)

        # if not os.path.exists(next_tensorboard_log):
        #     os.makedirs(next_tensorboard_log)

    def current_epoch_result_write_2_tensorboard(self,current_epoch_result,epoch):
        for key ,value in current_epoch_result.items():
            self.writer.add_scalar(key, value, epoch)
        # 将损失和准确率记录到 TensorBoard
        # self.writer.add_scalar('Loss/train', current_epoch_result["Loss"], epoch)
        # self.writer.add_scalar('Accuracy/train', current_epoch_result["Accuracy"], epoch)

    def write_dict_to_csv(self,filename, data_dict):
        # 写入字典到 CSV 文件
        with open(filename, 'a', newline='') as csvfile:
            fieldnames = list(data_dict.keys())
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            # 如果文件为空，写入表头
            if os.stat(filename).st_size == 0:
                writer.writeheader()

            # 写入数据
            writer.writerow(data_dict)
