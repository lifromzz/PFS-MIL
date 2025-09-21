from Student_model_train.club import train_test
from Student_model_train.club import args_base


def main():
    args=args_base.get_args(dataname="CPTAC")
    args.train_bs=65
    args.tbc="D:\\project\\Teacher_model_train\\TCGA-NSCLC-r101\\recoder\\CPTAC-ssl-l3\\checkpoint\\ProMIL_best_SSL_r50_l3"
    args.data = "D:\\Dataset\\CPTAC\\represention\\ssl_BT_r50_level1_224\\stage3"

    args.mothed="Teacher_bag"
    train_test.main(args,train=True,test="Single",)
    args.mothed = "Teacher_instance"
    train_test.main(args, train=False, test="Single")
    args.mothed = "Teacher_instance_bag"
    train_test.main(args, train=False, test="Single")


if __name__ == '__main__':
    main()