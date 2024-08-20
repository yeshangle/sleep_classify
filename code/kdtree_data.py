import pandas as pd
import numpy as np
import sys
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from tqdm import tqdm
import joblib
import os

# 1. 加载相关的数据
path = "../after_process_data/after_process_data/"
# 获取所有的不是motion得excel的文件
execl_files = [f for f in os.listdir(path) if not f.endswith("motion.xlsx")]
# 创建一个进度条
progress_bar = tqdm(total=len(execl_files), desc="progress file")
column_names = ['feature1', 'feature2', 'feature3', 'feature4', 'feature5', 'feature6', "label"]
# 创建一个空的DataFrame的数据
dataframes = list()
# 2 数据处理
for file_name in execl_files:
    file_path = os.path.join(path, file_name)
    df = pd.read_excel(file_path, header=None)
    if file_name.endswith("left.xlsx"):
        # 表示左躺睡觉
        df["label"] = 1
    elif file_name.endswith("m.xlsx"):
        # 表示正常睡觉
        df["label"] = 2
    else:
        # 表示右躺睡觉
        df["label"] = 3
    # 当前的文件追加到all_data中
    dataframes.append(df)
    progress_bar.update(1)
progress_bar.close()  # 完成时关闭进度条
new_df = pd.concat(dataframes, ignore_index=False)
new_df.label = new_df.label.astype(np.int32)
# 3. 根据需求获取最原始的特征属性矩阵X和目标属性Y
X = new_df.iloc[:, :-1]
# print(X.shape)
# Y = new_df.iloc[:, -1]

# # 4 数据分割
# x_train, x_test, y_train, y_test = train_test_split(X, Y, train_size=0.8)
#
# # 5 构建kdtree
# KD_tree = KNeighborsClassifier(n_neighbors=3, weights='uniform', algorithm='kd_tree')
#
# # 6. 模型的训练
# KD_tree.fit(x_train, y_train)
# # 7 模型的评估
# print("KD_tree算法：测试集上的效果(准确率):{}".format(KD_tree.score(x_test, y_test)))
# print("KD_tree算法：训练集上的效果(准确率):{}".format(KD_tree.score(x_train, y_train)))
# # 8 模型保存
# save_path = "../model/kd_tree.m"
# joblib.dump(KD_tree, save_path)



