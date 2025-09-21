from Student_model_train.club import train_test
from Student_model_train.club import args_base


def main():
    args=args_base.get_args(dataname="HCH")
    args.train_bs=64
    args.tbc="D:\\project\\Teacher_model_train\\TCGA-NSCLC-r101\\recoder\\HCH-ssl-l3\\checkpoint\\ProMIL_best_SSL_r50_l3"
    args.data = "D:\\HNSZL Train\\represention\\ssl_BT_r50_level2_224\\stage3"

    args.mothed="Teacher_bag"
    train_test.main(args,train=True,test=True)
    args.mothed = "Teacher_instance"
    train_test.main(args, train=False, test=True)
    args.mothed = "Teacher_instance_bag"
    train_test.main(args, train=False, test=True)



if __name__ == '__main__':
    main()