from Student_model_train.club import train_test
from Student_model_train.club import args_base


def main():
    args=args_base.get_args(dataname="TCGA-NSCLC")
    args.train_bs=64
    args.tbc="D:\\project\\Teacher_model_train\\TCGA-NSCLC-r101\\recoder\\TCGA-NSCLC-ssl-l3\\checkpoint\\ProMIL_best_ssl_r50_l3"
    args.data="D:\\Dataset\\TCGA-LUNG\\represention\\all_1042\\stage3"

    args.mothed="Teacher_bag"
    train_test.main(args,train=False,test=True)
    args.mothed = "Teacher_instance"
    train_test.main(args, train=False, test=True)
    args.mothed = "Teacher_instance_bag"
    train_test.main(args, train=False, test=True)

if __name__ == '__main__':
    main()