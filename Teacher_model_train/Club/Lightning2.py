import torch
import os
import tqdm
import numpy as np
import torch.nn.functional as F
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import roc_auc_score, f1_score, recall_score, precision_score, accuracy_score,roc_curve

def get_lable_dict():
    label_dict = {"LUAD": 0,"LUSC": 1,"SCLC": 2,"Normal": 3}
    return label_dict
def get_path(data_name,seed,fold,stage="stage3"):
    if data_name=="LUNG":
        return {
            "data":"D:\\HNSZL Train\\represention\\r50_level0_224\\"+stage,
            "split_path":"D:\\project\\laten_mamba_main\\data\\LUNG\\LUNG_file_labels.xlsx",
            "train_sheet":"train",
            "test_sheet":"test",
        }
    elif data_name=="CPTAC":
        split_path="D:\\project\\laten_mamba_main\\data\\CPTAC\\file_labels_GPTtxt_seed{}.xlsx".format(seed)
        return {
            "data": "D:\\Dataset\\CPTAC\\represention\\r50_level0_224\\"+stage,
            "split_path":split_path,
            "train_sheet":"train_fold{}".format(fold),
            "test_sheet":"test_fold{}".format(fold),
        }
    else:
        split_path = "D:\\project\\laten_mamba_main\\data\\TCGA\\file_labels_seed{}.xlsx".format(seed)
        return {
            "data": "D:\\Dataset\\TCGA-LUNG\\represention\\r50_level0_224\\"+stage,
            "split_path": split_path,
            "train_sheet": "train_fold{}".format(fold),
            "test_sheet": "test_fold{}".format(fold),
        }


class Lightning():
    def __init__(self,
                lg_logger,
                device="cuda",
                num_class=4):
        self.device=device
        self.nclass=num_class
        self.logger=lg_logger

        self.best_metric=0

        self.metric_logger={
            "Accuracy":accuracy_score,
            "AUC":roc_auc_score,
            "F1":f1_score,
            "Recall":recall_score,
            "Precision":precision_score,
        }

    def save_checkpoint(self,model,optimizer,current_result,epoch):
        #只保存训练集上准确率最高的模型
        current_epoch_accuracy=current_result["Accuracy"]
        if current_epoch_accuracy<self.best_metric:
            chekpoint_filename = "last_{}_{:.2f}.pth.tar".format(epoch, current_epoch_accuracy)
        else:
            self.best_metric = current_epoch_accuracy
            chekpoint_filename = "best_{}_{:.2f}.pth.tar".format(epoch, current_epoch_accuracy)
            self.checkpoint_path=os.path.join(self.logger.checkpoint_folder,chekpoint_filename)
            torch.save({
                "epoch":epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, self.checkpoint_path)

    def load_checkpoint(self,model,checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['state_dict'])
        return model

    def train(self,
              optimizer,
              model,
              criterion,
              train_dataloder,
              args):

        self.logger.next_tensorbard()
        for epoch in range(args.start_epoch,args.num_epochs):
            current_result=self.train_loop(optimizer,
                                           model,
                                           criterion,
                                           train_dataloder)
            self.logger.current_epoch_result_write_2_tensorboard(current_result,epoch)
            self.save_checkpoint(model,optimizer,current_result,epoch)

        # 关闭 SummaryWriter
        self.logger.writer.close()

    def train_loop(self,optimizer,model,criterion,train_dataloder):
        current_result={
            "Loss":0,
            "Accuracy":0
        }
        train_bar=tqdm.tqdm(train_dataloder,desc="Train")
        for x,target in train_bar:
            x=x.to(self.device)
            target=target.to(self.device)

            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            # 计算准确率
            _, predicted = torch.max(output.data, 1)
            accuracy = self.metric_logger["Accuracy"](target.cpu().numpy(), predicted.cpu().numpy())
            train_bar.set_postfix({"loss":loss.item(),"Accuracy":accuracy})

            current_result["loss"]+=loss.item()
            current_result["Accuracy"]+=accuracy
        current_result["loss"] /= len(train_bar)
        current_result["Accuracy"] /= len(train_bar)
        return current_result
    def test(self,
             model,
             test_dataloader,
             checkpoint_path=None,
             csv_path=None,
             stage="l3"):


        if checkpoint_path is not None:
            model=self.load_checkpoint(model,checkpoint_path)
        model.cuda()
        model.eval()

        # 初始化变量用于统计
        predictions = []
        pred_label = []
        labels_list = []

        test_bar=tqdm.tqdm(test_dataloader,desc="Testing")
        with torch.no_grad():
            for x,labels in test_bar:
                if isinstance(x,dict):
                    if stage in x.keys():
                        x=x[stage]
                    else:
                        x=x["l34"]
                        if stage == "l3":
                            x = x[:,:, :1024]
                        elif stage == "l4":
                            x = x[:,:, 1024:]
                if len(x.size()) > 2:
                    x = x.squeeze(0)

                    # if "l34" in x.keys() and (stage not in x.keys()):
                    #     if stage=="l3":
                    #         x=x["l34"][:,:1024]
                    #     elif stage=="l4":
                    #         x=x["l34"][:,1024:]
                    # else:
                    #     x=x[stage]

                    # if len(x.size())>2:
                        # print(x.size())
                        # x=x.squeeze(0)
                        # print(x.size())
                # if len(x.size()) > 2:
                #     x=x.squeeze(0)
                x = x.to(self.device)  # 如果有GPU，将数据移动到GPU上
                labels = labels.to(self.device)  # 如果有GPU，将数据移动到GPU上
                # 模型推断
                outputs = model(x)
                outputs = F.softmax(outputs)
                _, predicted = torch.max(outputs.data, 1)

                # 统计预测结果和真实标签
                predictions.extend(outputs.data.cpu().numpy())
                pred_label.extend(predicted.cpu().numpy())
                labels_list.extend(labels.cpu().numpy())
        test_score=self.save_reslut_to_csv(predictions,pred_label,labels_list,csv_path)
        return test_score
    def test2(self,
             model,
             test_dataloader,
             checkpoint_path=None,
             csv_path=None,
             stage="l3"):


        if checkpoint_path is not None:
            model=self.load_checkpoint(model,checkpoint_path)
        model.cuda()
        model.eval()

        # 初始化变量用于统计
        predictions = []
        pred_label = []
        labels_list = []

        test_bar=tqdm.tqdm(test_dataloader,desc="Testing")
        with torch.no_grad():
            for x,labels in test_bar:
                if isinstance(x,dict):
                    x=x[stage]
                    if len(x.size())>2:
                        # print(x.size())
                        x=x.squeeze(0)
                        # print(x.size())
                x = x.to(self.device)  # 如果有GPU，将数据移动到GPU上
                labels = labels.to(self.device)  # 如果有GPU，将数据移动到GPU上
                # 模型推断
                outputs,_ = model(x)
                outputs = F.softmax(outputs)
                _, predicted = torch.max(outputs.data, 1)

                # 统计预测结果和真实标签
                predictions.extend(outputs.data.cpu().numpy())
                pred_label.extend(predicted.cpu().numpy())
                labels_list.extend(labels.cpu().numpy())
        test_score=self.save_reslut_to_csv(predictions,pred_label,labels_list,csv_path)
        return test_score
    def save_reslut_to_csv(self,predictions,pred_label,labels_list,csv_path):
        test_score={}
        for name, metric in self.metric_logger.items():
            if name in ["F1","Recall","Precision"]:
                score = metric(labels_list, pred_label,average="micro")
            elif name=="AUC":
                if max(labels_list)>1:  #多分类AUC
                    auc_total=0
                    for c in set(labels_list):
                        c_score = self.roc_threshold([1 if label == c else 0 for label in labels_list],
                                         [1 if pred == c else 0 for pred in pred_label])
                        auc_total += c_score
                    score=auc_total/len(set(labels_list))
                else:  #二分类AUC
                    score=self.roc_threshold(labels_list,np.array(predictions)[:,-1])
            else:
                score = metric(labels_list, pred_label)
            test_score[name]=score
            print('{} on test set: {:.2f}%'.format(name, score*100))

        # 计算每个类别的评估指标
        for name, metric in self.metric_logger.items():
            for c in set(labels_list):
                c_score = metric([1 if label == c else 0 for label in labels_list],
                                                   [1 if pred == c else 0 for pred in pred_label])
                test_score["{}_{}".format(c,name)]=c_score
                print('Class {} {}: {:.2f}%'.format(c,name, c_score * 100))
        if csv_path is not None:
            self.logger.write_dict_to_csv(csv_path,test_score)
        return test_score
    def print_information(self,params):
        # print("训练集数据路径：",params.data)
        print("五折交叉excel表格路径：",params.split_path)
        print("excel表格中训练sheet为：",params.train_sheet)
        print("excel表格中测试sheet为：",params.test_sheet)
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("当前训练硬件为：",device)


    def roc_threshold(self, label,prediction,th=False):
        fpr, tpr, threshold = roc_curve(label, prediction, pos_label=1)
        fpr_optimal, tpr_optimal, threshold_optimal = self.optimal_thresh(fpr, tpr, threshold)
        c_auc = roc_auc_score(label, prediction)
        if th:
            return c_auc, threshold_optimal
        else:
            return c_auc

    def optimal_thresh(self,fpr, tpr, thresholds, p=0):
        loss = (fpr - tpr) - p * tpr / (fpr + tpr + 1)
        idx = np.argmin(loss, axis=0)
        return fpr[idx], tpr[idx], thresholds[idx]