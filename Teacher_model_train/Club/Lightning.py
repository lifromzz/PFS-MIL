import os
import time
import torchmetrics
import tqdm
import torch
import warnings


from utils import csv_utils

warnings.filterwarnings("ignore")  # 忽略警告信息

class LightningModule():
    def __init__(self,
                 nclass
                 ):
        super().__init__()
        self.nclass=nclass

        self.task="multiclass" if self.nclass>2 else "binary"

        self.macro_AUC = torchmetrics.AUROC(num_classes=self.nclass, average='macro',task=self.task)
        self.micro_AUC=torchmetrics.AUROC(num_classes=self.nclass,average="none",task=self.task)
        self.macro_metrics = torchmetrics.MetricCollection([
            torchmetrics.Accuracy(average='micro', num_classes=self.nclass,task=self.task),
            torchmetrics.F1Score(average='macro', num_classes=self.nclass,task=self.task),
            torchmetrics.Recall(average='macro', num_classes=self.nclass,task=self.task),
            torchmetrics.Precision(average='macro', num_classes=self.nclass,task=self.task),
            torchmetrics.Specificity(average='macro', num_classes=self.nclass,task=self.task),
            torchmetrics.CohenKappa(num_classes=self.nclass,task=self.task), ])

        self.none_metrics = torchmetrics.MetricCollection(
            [torchmetrics.Accuracy(average='none', num_classes=self.nclass,task=self.task),
             torchmetrics.F1Score(average='none', num_classes=self.nclass,task=self.task),
             torchmetrics.Recall(average='none', num_classes=self.nclass,task=self.task),
             torchmetrics.Precision(average='none', num_classes=self.nclass,task=self.task),
             torchmetrics.Specificity(average='none', num_classes=self.nclass,task=self.task),
             ])

    def load_checkpoint(self, model, checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['state_dict'])
        return checkpoint["epoch"]

    def test(self,model,testloader,checkpoint_files,save,csv_path):
        if checkpoint_files is not None:
            epoch=self.load_checkpoint(model,checkpoint_files)
        target,probs=self.validate_loop(model,testloader,epoch)
        self.eval_result(probs,target,save=save,csv=csv_path)


    def validate_loop(self,model, val_loader,epoch):
        model.cuda()
        model.eval()
        prob_list, target_list, feat_list = [], [], []
        val_loader = tqdm.tqdm(val_loader, "val:{}".format(epoch))

        with torch.no_grad():
            for data in val_loader:
                x, target, batch = data.x, data.y, data.batch
                x = x.cuda()
                batch = batch.cuda()
                target = target.cuda()

                prob = model(x, batch, False)

                prob_list.append(prob.cpu().detach())
                target_list.extend(target)
                #feat_list.extend(feat)
        return torch.tensor(target_list), torch.cat(prob_list)

    def eval_result(self,prob,target,save,csv):
        pred = torch.argmax(prob, -1)
        # pred, prob = self.metric_test(mothed, train_target, train_feat, val_prob, val_feat)

        metric_list=[]
        auc = self.macro_AUC(prob[:,-1], target)
        print("AUC:{}".format(auc))
        metric_list.append(auc)

        metrics = self.macro_metrics(pred, target)
        for key, value in metrics.items():
            print("{}:{}".format(key, value))
            metric_list.append(value)

        # csv_utils.write_csv(csv,"{},{}\n".format(metrics["Accuracy"],auc),"a")

        if len(prob.shape)>2:
            metrics = self.none_metrics(pred, target)
            for key, value in metrics.items():
                print("{}:{}".format(key, value))
                for v in value:
                    metric_list.append(v)
        if save:
            input="{},"*(len(metric_list)-1)+"{}\n"
            csv_utils.write_csv(csv, input.format(*metric_list), "a")


