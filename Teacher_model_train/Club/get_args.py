import argparse
def get_args():
    parser = argparse.ArgumentParser(description='abc')
    parser.add_argument('--EPOCH', default=300, type=int)
    parser.add_argument('--epoch_step', default='[100]', type=str)
    parser.add_argument('--device', default='cuda', type=str)

    parser.add_argument('--lr', default=1e-1, type=float)
    parser.add_argument('--weight_decay', default=1e-5, type=float)
    parser.add_argument("--m",default=0.4,type=float)
    parser.add_argument("--s",default=64,type=int)
    parser.add_argument("--bs",default=16,type=int)
    parser.add_argument('--num_workers', default=0, type=int)
    parser.add_argument('--nclass', default=2, type=int)
    parser.add_argument("--input_dim",default=3072,type=int)
    parser.add_argument('--mDim', default=1536, type=int)

    parser.add_argument('--dataset_name', default='TCGA-NSCLC', type=str)
    parser.add_argument("--processed_dir",default="processed_l34",type=str)
    parser.add_argument("--recoder",default="./recoder",type=str)
    args = parser.parse_args()
    return args