import argparse
import torch
def get_args(dataname):
    numclass={
        "TCGA-NSCLC":2,
        "CAMELYON":2,
        "HCH":4,
        "CPTAC":3
    }
    argp = argparse.ArgumentParser()
    argp.add_argument("--model_name", default="ProMIL")
    argp.add_argument("--dataname", default=dataname,type=str)
    argp.add_argument("--stage", default="stage3", type=str)
    argp.add_argument("--loss_fn",default="amsoftmax",type=str)

    argp.add_argument("--nclass", default=numclass[dataname], type=int)
    argp.add_argument("--T_input_dim",default=1024,type=int)
    argp.add_argument("--T_embed_dim", default=512, type=int)

    argp.add_argument("--input_dim", default=1024, type=int)
    argp.add_argument("--embed_dim", default=512, type=int)
    argp.add_argument("--attn_dropout",default=0.3,type=float)
    argp.add_argument("--attn_emb",default=128,type=float)

    argp.add_argument("--proxy",default=True,type=bool)
    argp.add_argument("--mothed",default='bag',choices=['bag','instance',"instance_bag"])

    # 实例比例
    argp.add_argument("--top_per", default=0.01, type=float)
    argp.add_argument("--bottom_per", default=0.01, type=float)

    # 损失函数
    argp.add_argument("--bag_m", default=0.4, type=float)
    argp.add_argument("--bag_s", default=64, type=int)
    argp.add_argument("--instance_m", default=0.4, type=float)
    argp.add_argument("--instance_s", default=64, type=int)

    argp.add_argument("--train_bs", default=64)
    argp.add_argument("--test_bs", default=1)

    argp.add_argument("--num_workers", default=4)
    argp.add_argument("--lr", default=1e-2)
    argp.add_argument("--wd", default=1e-5)
    argp.add_argument("--start_epoch", default=0, type=int)
    argp.add_argument("--num_epochs", default=200, type=int)
    argp.add_argument("--start_seed",default=0,type=int)
    argp.add_argument("--end_seed",default=20,type=int)
    argp.add_argument("--epoch_frq",default=25,type=str)

    argp.add_argument("--tensorboard_dir", default="./recoder/{}/log".format(dataname))
    argp.add_argument("--checkpoint_dir", default="./recoder/{}/checkpoint".format(dataname))
    argp.add_argument("--metic_dir", default="./recoder/{}/results".format(dataname))
    argp.add_argument("--metric",default="loss")
    argp.add_argument("--patient", default=20)
    argp.add_argument("--seed", default=0)
    argp.add_argument("--device", default="cuda")
    args = argp.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.device = device
    return args