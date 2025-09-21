import numpy as np
import torchmetrics
from sklearn.metrics import roc_auc_score, f1_score, recall_score, precision_score, accuracy_score,roc_curve
import csv
import os
import torch

class metric_lg():
    def __init__(self,metric_dir):
        self.metric_logger = {
            "Accuracy": accuracy_score,
            "AUC":self.roc_threshold,
            "F1": f1_score,
            "Recall": recall_score,
            "Precision": precision_score,
        }
        self.metric_dir=metric_dir
    def init_metric_dir(self):
        if not os.path.exists(self.metric_dir):
            os.makedirs(self.metric_dir)
    def Accuracy(self,predictions,targets):
        # 获取预测结果中概率最大的类别作为预测类别
        _, predicted_labels = torch.max(predictions, 1)

        # 比较预测类别与真实标签，统计预测正确的数量
        correct = (predicted_labels == targets).sum().item()

        # 计算正确率
        accuracy = correct / targets.size(0)

        return accuracy

    def get_reslut(self,epoch,predictions,pred_label,labels_list,csv_path=None):

        test_score={}
        for name, metric in self.metric_logger.items():
            if name in ["F1","Recall","Precision"]:
                score = metric(labels_list, pred_label,average="micro")
            elif name=="AUC":
                if max(labels_list)>1:  #多分类AUC
                    auc_total=0
                    for c in set(labels_list):
                        c_score = metric([1 if label == c else 0 for label in labels_list],
                                         [1 if pred == c else 0 for pred in pred_label])
                        auc_total += c_score
                    score=auc_total/len(set(labels_list))
                else:  #二分类AUC
                    score=self.roc_threshold(labels_list,predictions[:,-1])
                    # score=metric(labels_list,predictions[:,-1])
                    # score = metric(labels_list, pred_label)
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

        # test_score=self.eval_metric(predictions[:,-1],labels_list)
        if csv_path is not None:
            w_a="w" if epoch==0 else "a"
            self.write_dict_to_csv(csv_path,test_score,w_a)
        return test_score

    def write_dict_to_csv(self,filename, data_dict,w_a):
        self.init_metric_dir()
        # 写入字典到 CSV 文件
        with open(filename, w_a, newline='') as csvfile:
            fieldnames = list(data_dict.keys())
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            # 如果文件为空，写入表头
            if os.stat(filename).st_size == 0:
                writer.writeheader()

            # 写入数据
            writer.writerow(data_dict)

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

    def eval_metric(self,oprob, label):

        auc, threshold = self.roc_threshold(label,oprob)
        prob = oprob > threshold
        label = label > threshold

        TP = (prob & label).sum(0)
        TN = ((~prob) & (~label)).sum(0)
        FP = (prob & (~label)).sum(0)
        FN = ((~prob) & label).sum(0)

        accuracy = np.mean((TP + TN) / (TP + TN + FP + FN + 1e-12))
        precision = np.mean(TP / (TP + FP + 1e-12))
        recall = np.mean(TP / (TP + FN + 1e-12))
        specificity = np.mean(TN / (TN + FP + 1e-12))
        F1 = 2 * (precision * recall) / (precision + recall + 1e-12)

        Tsocre = {"Accuracy":accuracy,
                  "AUC":auc,
                  "F1":F1,
                  "Recall":recall,
                  "Precision":precision,
                  "specificity":specificity}
        return Tsocre