import os
import pandas as pd

label_dict = {"LUAD": 0, "LUSC": 1, "SCLC": 3, "Normal": 2}
# 读取文件中的所有数据并提取文件名
data_dir = ["D:\\HNSZL Train\\LUAD",
            "D:\\HNSZL Train\\LUSC",
            "D:\\HNSZL Train\\Normal",
            "D:\\HNSZL Train\\SCLC"]
# data_dir = ["D:\\HNSZL Train\\LUAD",
#             "D:\\HNSZL Train\\LUSC",]
repre_dir="D:\\HNSZL Train\\represention\\r50_level2_224\\stage3"
pt_filenames=[filename.split(".")[0] for filename in os.listdir(repre_dir)]

total_slide_names=[]
total_slide_labels=[]
for data_path in data_dir:
    class_base_name=os.path.basename(data_path)
    class_label_int=label_dict[class_base_name]
    filenames=[filename for filename in os.listdir(data_path) if "mrxs" not in filename]
    filenames=[filename for filename in filenames if filename in pt_filenames]

    total_slide_names.extend(filenames)
    total_slide_labels.extend([class_label_int]*len(filenames))


test_data_path="D:\\HNSZL Test"
test_filename=os.listdir(test_data_path)
# 将文件名和标签写入到csv文件中
train_df = pd.DataFrame({'file_name': total_slide_names, 'label': total_slide_labels})
test_df = pd.DataFrame({'file_name': test_filename})

with pd.ExcelWriter('../data/LUNG/Lung_file_labels_new.xlsx') as writer:
    train_df.to_excel(writer, sheet_name='train', index=False)
    test_df.to_excel(writer, sheet_name='test', index=False)

