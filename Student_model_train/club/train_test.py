import os.path
from Student_model_train.club.Lightning_stage2 import Lightning
from Student_model_train.club import utils
from Student_model_train.model.stage2_model_linerfc import stage2_model
from Student_model_train.model.Teacher import Teacher_model


def set_data(args,csv_seed,csv_fold,test):
    if args.dataname=="TCGA-NSCLC":
        args.excel_path = "D:\\Dataset\\TCGA-LUNG\\represention\\all_1042\\file_labels_seed{}.xlsx".format(csv_seed)
    elif args.dataname=="CPTAC":
        args.excel_path = "D:\\project\\laten_mamba_main\\data\\CPTAC\\file_labels_GPTtxt_seed{}.xlsx".format(csv_seed)
    elif args.dataname=="HCH":
        args.excel_path = "D:\\project\\laten_mamba_main\\data\\LUNG\\file_labels_seed{}.xlsx".format(csv_seed)
    elif args.dataname=="zzz":
        args.excel_path = "E:\\BaiduNetdiskDownload\\zzztest\\data.csv"

    args.stage1_train_best_path= args.teacher_best_ckp = "{}_seed{}.pth".format(args.tbc, args.seed)
    args.train_sheet = "train_fold{}".format(csv_fold)
    args.test_sheet = "test_fold{}".format(csv_fold)
    args.train_csv_path = "{}/train_on_kflod_{}_{}.csv".format(args.metic_dir, args.dataname, args.seed)

    args.test_csv_path = "{}/re_test_on_kflod_{}_{}.csv".format(args.metic_dir, args.dataname, args.mothed)

def main(args,train,test):
    args.des = "stage2_{}_{}_{}".format(args.embed_dim, args.bottom_per * 100, args.train_bs)

    for i in range(args.start_seed,args.end_seed):
        utils.fix_random_seed(i)
        csv_seed, csv_fold = i // 5, i % 5
        args.seed=i
        set_data(args,csv_seed, csv_fold,test)

        stage1_teacher=Teacher_model(input_dim=args.T_input_dim,
                             embed_dim=args.T_embed_dim,
                             n_class=args.nclass)

        model=stage2_model(input_dim=args.input_dim,
                          embed_dim=args.embed_dim,
                          AD=args.attn_dropout,
                          AL=args.attn_emb,
                          n_class=args.nclass
                          )

        lg = Lightning(args)
        if train:
            lg.train(stage1_model=stage1_teacher,
                     stage2_model=model,
                     stage1_best_ckp=args.stage1_train_best_path,
                     )

        checkpoint_path = os.path.join(args.checkpoint_dir,
                                       "{}_best_seed{}.pth".format(args.des, args.seed))
        if not os.path.exists(checkpoint_path):
            checkpoint_path = os.path.join(args.checkpoint_dir,
                                           "{}_epoch0_seed{}.pth".format(args.des, args.seed))

        if test:
            test_epoch = 0 if i == 0 else 200
            lg.test(epoch=test_epoch,
                    stage1_model=stage1_teacher,
                    stage2_model=model,
                    stage1_checkpoint_path=args.stage1_train_best_path,
                    stage2_checkpoint_path=checkpoint_path,
                    csv_path=args.test_csv_path)
