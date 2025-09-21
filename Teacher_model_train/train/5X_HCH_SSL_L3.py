import os.path
import argparse
import torch
import tqdm
import numpy as np

from sklearn.metrics import accuracy_score
from ProMIL_module import ProMIL
from EmbededFeatsDataset_proMIL import EmbededFeatsDataset
from EmbededFeatsDataset_proMIL import EmbededFeatsDataset_ProMIL
# torch.autograd.set_detect_anomaly(True)
from Teacher_model_train.Club.Lightning2 import Lightning
from Teacher_model_train.Club.logger import logger
from Teacher_model_train.Club.stop_early_LG import stop_early_lg
from ranger import Ranger
from Loss_hub import AMSoftmax

parser = argparse.ArgumentParser(description='abc')
parser.add_argument('--EPOCH', default=300, type=int)
parser.add_argument('--epoch_step', default='[100]', type=str)
parser.add_argument('--device', default='cuda', type=str)

parser.add_argument('--lr', default=1e-1, type=float)
parser.add_argument('--weight_decay', default=1e-5, type=float)
parser.add_argument("--m",default=0.4,type=float)
parser.add_argument("--s",default=64,type=int)
parser.add_argument('--num_workers', default=4, type=int)
parser.add_argument('--num_cls', default=4, type=int)
parser.add_argument("--input_dim",default=1024,type=int)
parser.add_argument('--mDim', default=512, type=int)

parser.add_argument("--recoder",default="./recoder",type=str)
args = parser.parse_args()



def test(lg):

    testset = EmbededFeatsDataset(args.data, split_path=args.split_path, sheet_name=args.test_sheet)
    testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, drop_last=False)

    model=ProMIL(n_class=args.num_cls,input_dim=args.input_dim,embed_dim=args.mDim)

    checkpoint_files = os.path.join(args.recoder, args.dataset_name,"checkpoint", args.ckp)
    lg.test(model,testloader,checkpoint_files,args.csv_path,"l34")



def train():
    stlg=stop_early_lg(patient=20,metric="acc")
    trainset = EmbededFeatsDataset_ProMIL(args.data,split_path=args.split_path,
                                          sheet_name=args.train_sheet,batch_size=16,
                                          stage=False)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=1, shuffle=True, drop_last=False,num_workers=args.num_workers)

    model=ProMIL(n_class=args.num_cls,input_dim=args.input_dim,embed_dim=args.mDim)
    model.cuda()
    model.train()

    optimizer0 = Ranger(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    ce_cri = AMSoftmax(m=args.m,s=args.s)

    best_acc = 0.1

    #logger_lg.next_tensorbard()

    for ii in range(args.EPOCH):

        current_result = {
            "Loss": 0,
            "Accuracy": 0
        }

        for param_group in optimizer0.param_groups:
            curLR = param_group['lr']
            print('current learning rate {}'.format(curLR))

        train_bar=tqdm.tqdm(trainloader,desc="Training {}".format(ii))
        for i, (inputs, labels,batch) in enumerate(train_bar):
            inputs = torch.squeeze(inputs)
            labels = torch.squeeze(labels)
            batch = torch.squeeze(batch)
            inputs_tensor = inputs.to(args.device)
            labels = labels.to(args.device)
            batch = batch.to(args.device)

            tPredict=model(inputs_tensor,batch,True)

            loss0 = ce_cri(tPredict, labels)
            optimizer0.zero_grad()
            loss0.backward()
            optimizer0.step()

            # 计算准确率
            _, predicted = torch.max(tPredict.data, 1)
            accuracy = accuracy_score(labels.cpu().detach().numpy(),predicted.cpu().detach().numpy())
            train_bar.set_postfix({"loss": loss0.item(), "Accuracy": accuracy})

            current_result["Loss"] += loss0.item()
            current_result["Accuracy"] += accuracy
            train_bar.set_postfix({"loss":loss0.item(),"Accuracy":accuracy})

        current_result["Loss"] /= len(train_bar)
        current_result["Accuracy"] /= len(train_bar)


        #logger_lg.current_epoch_result_write_2_tensorboard(current_result,ii)
        if stlg.stop(current_result["Accuracy"]):
            break
        # auc, acc, f1 = TestModel(valloader)
        if current_result["Accuracy"] > best_acc:
            best_acc = current_result["Accuracy"]
            tsave_dict = {
                'state_dict': model.state_dict(),
                "optimizer":optimizer0.state_dict(),
                "acc":current_result["Accuracy"],
                "epoch":ii
            }
            save_path=os.path.join(args.recoder,args.dataset_name,"checkpoint",args.ckp)
            torch.save(tsave_dict, save_path)
            print('new best auc. saved to .',save_path)


def main():
    args.dataset_name="HCH-ssl-l3"
    logger_lg = logger(dataset_name=args.dataset_name, log_root=args.recoder)
    lg = Lightning(lg_logger=logger_lg,num_class=args.num_cls)
    for i in range(0,20):
        args.seed = i
        csv_seed, csv_fold = i // 5, i % 5

        args.data = "D:\\HNSZL Train\\represention\\ssl_BT_r50_level2_224\\stage3"
        args.split_path = "D:\\project\\laten_mamba_main\\data\\LUNG\\file_labels_seed{}.xlsx".format(csv_seed)

        args.train_sheet = "train_fold{}".format(csv_fold)
        args.test_sheet = "test_fold{}".format(csv_fold)
        args.csv_path = "./recoder/{}/results/train_on_kflod.csv".format(args.dataset_name)
        args.ckp = "ProMIL_best_SSL_r50_l3_seed{}.pth".format(args.seed)


        # train()
        test(lg)


if __name__ == '__main__':
    main()