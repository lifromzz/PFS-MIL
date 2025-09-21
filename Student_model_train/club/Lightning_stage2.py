import numpy as np
import torch
import os
import tqdm
import collections
from torch.utils.data import DataLoader
from Student_model_train.dataset.stage2_dataset import stage2_dataset
from Student_model_train.dataset.EmbededFeatsDataset import myDatasetInMemory

from Student_model_train.club.Tensorboard_LG import tensorboard_lg
from Student_model_train.club.checkpoint_LG import checkpoint_lg
from Student_model_train.club.metric_LG import metric_lg
from Student_model_train.club.stop_early_LG import stop_early_lg
from Student_model_train.club.ranger import Ranger
from Student_model_train.club.Loss_hub import AMSoftmax
from torch.nn import CrossEntropyLoss
from collections import Counter

class Lightning():
    def __init__(self,
                 args
                ):
        self.melg=metric_lg(metric_dir=args.metic_dir)
        self.stlg=stop_early_lg(metric=args.metric, patient=args.patient)
        self.tlg=tensorboard_lg(tensorboard_folder=args.tensorboard_dir)
        self.cklg=checkpoint_lg(metric=args.metric, checkpoint_dir=args.checkpoint_dir)
        self.args=args

    def get_loader(self,train,model,data,stage1_train_best_path,stage="l3"):
        sheet=self.args.train_sheet if train else self.args.test_sheet
        bs=self.args.train_bs if train else self.args.test_bs
        need_shuffle=train
        run_dataset = stage2_dataset(data,
                                     self.args.excel_path,
                                     sheet,
                                     model,
                                     stage1_train_best_path,
                                     bs,
                                     self.args.top_per,
                                     self.args.bottom_per,
                                     need_shuffle,
                                     stage)
        return run_dataset

    def train(self,
              stage2_model,
              stage1_model,
              stage1_best_ckp,
              testloader=False
              ):

        optimizer = Ranger(stage2_model.parameters(), lr=self.args.lr, weight_decay=self.args.wd)
        if self.args.loss_fn=="amsoftmax":
            bag_loss_fn = AMSoftmax(self.args.bag_m, self.args.bag_s)
            instance_loss_fn = AMSoftmax(self.args.instance_m, self.args.instance_s)
        elif self.args.loss_fn=="ce":
            bag_loss_fn = CrossEntropyLoss()
            instance_loss_fn = CrossEntropyLoss()

        train_dataloder=self.get_loader(train=True,model=stage1_model,data=self.args.data,stage1_train_best_path=stage1_best_ckp)
        if testloader:
            test_dataloader= self.get_loader(train=False,model=stage1_model,stage1_train_best_path=stage1_best_ckp)

        stage2_model.train()
        stage2_model.to(self.args.device)

        self.tlg.init_tensorbard(self.args.seed)
        self.tlg.mini_batch_count=train_dataloder.train_step

        for epoch in range(self.args.start_epoch,self.args.num_epochs):
            current_result = {
                "Bag_loss": 0,
                "instance_loss": 0,
                "total_loss":0,
                "Bag_acc":0,
                "Instance_acc":0
            }
            train_bar = tqdm.tqdm(train_dataloder.get_random_indexs(), desc="seed {} Training {}".format(self.args.seed,epoch))

            for train_index in train_bar:
                Hight_value_groups_x, \
                Hight_value_groups_y, \
                Compex_value_groups_x, \
                Compex_value_groups_y, \
                batch=train_dataloder.get_batch(train_index)

                Hight_value_groups_x=Hight_value_groups_x.to(self.args.device)
                Hight_value_groups_y=Hight_value_groups_y.to(self.args.device)
                Compex_value_groups_x=Compex_value_groups_x.to(self.args.device)
                Compex_value_groups_y=Compex_value_groups_y.to(self.args.device)
                batch=batch.to(self.args.device)

                instance_pro,bag_prob = stage2_model(Hight_value_groups_x,Compex_value_groups_x,batch,True)
                if (bag_prob is not None) and (instance_pro is not None):
                    bag_loss = bag_loss_fn(bag_prob, Compex_value_groups_y)
                    instance_loss = instance_loss_fn(instance_pro, Hight_value_groups_y)
                    total_loss=instance_loss+bag_loss
                elif (bag_prob is not None) and (instance_pro is None):
                    bag_loss = bag_loss_fn(bag_prob, Compex_value_groups_y)
                    total_loss= bag_loss
                else:
                    instance_loss=instance_loss_fn(instance_pro, Hight_value_groups_y)
                    total_loss = instance_loss
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

                if instance_pro is not None:
                    instance_acc=self.melg.Accuracy(instance_pro,Hight_value_groups_y)
                    current_result["Instance_acc"]=current_result["Instance_acc"]+instance_acc
                    current_result["instance_loss"]=current_result["instance_loss"]+instance_loss.item()
                if bag_prob is not None:
                    bag_acc = self.melg.Accuracy(bag_prob, Compex_value_groups_y)
                    current_result["Bag_acc"]=current_result["Bag_acc"]+bag_acc
                    current_result["Bag_loss"]=current_result["Bag_loss"]+bag_loss.item()
                current_result["total_loss"]=current_result["total_loss"]+total_loss.item()

                self.tlg.recoder_result(current_result)

            self.tlg.refresh_log(epoch=epoch)

            # 保存模型
            self.cklg.save_epoch_checkpoint(
                epoch_metric=self.tlg.recoder_dict["total_loss"],
                model=stage2_model,
                optimizer=optimizer,
                epoch=epoch,
                ckp_dir=self.cklg.checkpoint_dir,
                name=self.args.des,
                epoch_frq=self.args.epoch_frq,
                seed=self.args.seed)


            if testloader:
                self.test(epoch,stage1_model,stage2_model,test_dataloader=test_dataloader,csv_path=self.args.train_csv_path)

            if self.stlg.stop(self.tlg.recoder_dict["total_loss"]):
                break
    def test_preditc(self,
                     stage1_model,
                     stage2_model,
                     stage1_checkpoint_path=None,
                     stage2_checkpoint_path=None,
                     data=None,
                     stage="l3"
                     ):
        if stage2_checkpoint_path is not None:
            stage2_model = self.cklg.load_checkpoint(model=stage2_model, path=stage2_checkpoint_path)
        test_dataloader = self.get_loader(train=False, model=stage1_model,
                                              stage1_train_best_path=stage1_checkpoint_path,
                                          data=data,stage=stage)

        stage2_model.to("cuda")
        stage2_model.eval()
        # 初始化变量用于统计
        predictions = []
        pred_labels = []
        true_labels = []
        test_bar = tqdm.tqdm(test_dataloader.get_random_indexs(), desc="Testing")
        with torch.no_grad():
            for test_index in test_bar:
                if len(test_index) == 0:
                    continue
                Hight_value_groups_x, \
                    Hight_value_groups_y, \
                    Compex_value_groups_x, \
                    Compex_value_groups_y, \
                    batch = test_dataloader.get_batch(test_index)
                if len(Hight_value_groups_x) == 0:
                    continue

                Hight_value_groups_x = Hight_value_groups_x.to(self.args.device)
                Compex_value_groups_x = Compex_value_groups_x.to(self.args.device)
                batch = batch.to(self.args.device)

                instance_pro, bag_prob = stage2_model(Hight_value_groups_x, Compex_value_groups_x, batch, True)
                if ("instance" in self.args.mothed) and ("bag" not in self.args.mothed) and \
                        instance_pro is None:  # 防止实例为空
                    continue
                bag_prob, pred_label = self.test_mothed(instance_pro, bag_prob, self.args.mothed)

                predictions.extend(bag_prob.data.cpu().numpy())
                pred_labels.extend(pred_label.cpu().numpy())
                true_labels.extend(Compex_value_groups_y)

        true_labels = torch.tensor(true_labels).numpy()
        predictions = np.array(predictions)
        pred_labels = np.array(pred_labels)
        return predictions,true_labels,pred_labels

    def test(self,
             epoch,
             stage1_model,
             stage2_model,
             stage1_checkpoint_path=None,
             stage2_checkpoint_path=None,
             csv_path=None
             ):

        predictions,true_labels ,pred_labels= self.test_preditc(stage1_model,stage2_model,
             stage1_checkpoint_path,stage2_checkpoint_path,data=self.args.data)
        test_score=self.melg.get_reslut(epoch,predictions,pred_labels,true_labels,csv_path)
        return test_score
    def test_coop(self,
                  epoch,
                  stage1_model_SSL,
                  stage2_model_SSL,
                  stage1_SSL_checkpoint_path=None,
                  stage2_SSL_checkpoint_path=None,
                  SSL_data=None,
                  two_model=None,
                  three_model=None,
                  four_model=None,
                  csv_path=None
                  ):
        self.args.mothed="instance_bag"
        predictions_SSL, true_labels_SSL, pred_labels_SSL = self.test_preditc(stage1_model_SSL, stage2_model_SSL,
                                                                  stage1_SSL_checkpoint_path, stage2_SSL_checkpoint_path,
                                                                          SSL_data)
        test_score=self.melg.get_reslut(epoch,predictions_SSL,pred_labels_SSL,true_labels_SSL)


        predictions_two, true_labels_two, pred_labels_two = self.test_preditc(two_model["stage1"],
                                                                                             two_model["stage2"],
                                                                                             two_model["stage1_ckp"],
                                                                                             two_model["stage2_ckp"],
                                                                                             two_model["data"],
                                                                              two_model["stage"])
        test_score=self.melg.get_reslut(epoch,predictions_two,pred_labels_two,true_labels_two)

        predictions_three, true_labels_three, pred_labels_three= self.test_preditc(three_model["stage1"],
                                                                                             three_model["stage2"],
                                                                                             three_model["stage1_ckp"],
                                                                                             three_model["stage2_ckp"],
                                                                                             three_model["data"])
        test_score = self.melg.get_reslut(epoch, predictions_three, pred_labels_three, true_labels_three)

        predictions_four, true_labels_four, pred_labels_four = self.test_preditc(four_model["stage1"],
                                                                                    four_model["stage2"],
                                                                                    four_model["stage1_ckp"],
                                                                                    four_model["stage2_ckp"],
                                                                                    four_model["data"],
                                                                                 four_model["stage"])
        test_score = self.melg.get_reslut(epoch, predictions_three, pred_labels_three, true_labels_three)

        # predictions = (0.3*predictions_ImageNet + 0.7*predictions_SSL)
        pred_labels=self.majority_voting(pred_labels_SSL, pred_labels_two, pred_labels_three,pred_labels_four)
        # pred_labels = np.argmax(predictions,axis=1)

        test_score = self.melg.get_reslut(epoch, pred_labels, pred_labels, true_labels_SSL, csv_path)
        return test_score

    def majority_voting(self,predictions1, predictions2, predictions3,predictions4):
        # 确保所有预测数组的长度相同
        if not (len(predictions1) == len(predictions2) == len(predictions3)):
            raise ValueError("所有分类器的预测结果长度必须相同")

        majority_votes = []

        # 对每个样本进行投票
        for i in range(len(predictions1)):
            # 创建一个计数器来记录每个类别的票数
            vote = Counter()
            vote[predictions1[i]] += 1
            vote[predictions2[i]] += 1
            vote[predictions3[i]] += 1
            vote[predictions4[i]] += 1

            # 找出票数最多的类别
            majority_votes.append(vote.most_common(1)[0][0])

        return majority_votes
    def test_mothed(self,instance_pro,bag_prob,mothed):
        if "instance_bag" in mothed:
            if instance_pro is not None:
                instance_pro=torch.mean(instance_pro,dim=0,keepdim=True)
                bag_prob=(instance_pro+bag_prob)/2
        elif "instance" in mothed:
            # instance_pro=self.topk(instance_pro)
            bag_prob = torch.mean(instance_pro, dim=0, keepdim=True)
        pred_label=torch.argmax(bag_prob,dim=1)
        # bag_prob=torch.nn.functional.softmax(bag_prob)
        return bag_prob,pred_label
    def topk(self,predictions,k=20):
        # 计算每个预测的概率之和
        probabilities,_ = torch.max(predictions, dim=1)

        # 对概率进行排序，并获取前 20 个索引
        top_k_indices = torch.argsort(probabilities)[-k:]

        # 根据索引获取前 20 个概率最高的预测结果
        top_k_predictions = predictions[top_k_indices]
        return top_k_predictions


    def print_information(self,params):
        print("训练集数据路径：",params.data)
        print("五折交叉excel表格路径：",params.split_path)
        print("excel表格中训练sheet为：",params.train_sheet)
        print("excel表格中测试sheet为：",params.test_sheet)
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("当前训练硬件为：",device)


