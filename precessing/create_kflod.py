import pandas as pd
from sklearn.model_selection import StratifiedKFold

seed=4
xlsx_path="D:\\project\\laten_mamba_main\\data\\LUNG\\file_labels_seed{}.xlsx".format(seed)
# 读取xlsx文件
df = pd.read_excel("D:\\project\\laten_mamba_main\\data\\LUNG\\Lung_file_labels_new.xlsx", sheet_name="train")

# 将filename列和label列分别作为特征和标签
X = df["file_name"]
y = df["label"]

# 初始化五折交叉验证
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

# 进行五折交叉分割
for fold, (train_index, test_index) in enumerate(skf.split(X, y)):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    # 将文件名和标签写入到csv文件中
    train_df = pd.DataFrame({'file_name': X_train, 'label': y_train})
    test_df = pd.DataFrame({'file_name': X_test, 'label': y_test})
    if fold == 0:
        mode = "w"
    else:
        mode = "a"
    with pd.ExcelWriter(xlsx_path, mode=mode) as writer:
        train_df.to_excel(writer, sheet_name='train_fold{}'.format(fold), index=False)
        test_df.to_excel(writer, sheet_name='test_fold{}'.format(fold), index=False)
