import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
sns.set(font="Yu Gothic")


# train.csvはメモリ不足で開けないので、csvファイルを分割する
line_max_num = 60000
line_num = 1
file_num = 1
with open("./train.csv", "r", encoding="utf-8") as in_file:
    line = in_file.readline()
    while line:
        if line_num > line_max_num:
            line_num = 1
            file_num += 1
        else:
            with open("./train/train_{a:02}.csv".format(a=file_num), "a", encoding="utf-8") as out_file:
                out_file.write(line)
                line_num += 1
                line = in_file.readline()

# 分割csvファイルの読み込みで、変数名を動的に扱えるexec文を使ってみたけれど、IDE側で変数名を認識してくれないので、コーディング面を踏まえてこの方法は控える
for i in range(2, 13):
  exec("train_{a:02} = pd.read_csv('./train/train_{b:02}.csv', names=columns_name, encoding='utf-8')".format(a=i, b=i))

# 分割したcsvファイルは1データずつのズレがあるので、それを修正する形で分割csvファイルを再作成
train_01 = pd.read_csv("./train/train_01.csv",
                       encoding="utf-8")
columns_name = train_01.column
train_02 = pd.read_csv("./train/train_02.csv",
                       names=columns_name,
                       encoding="utf-8")
train_03 = pd.read_csv("./train/train_03.csv",
                       names=columns_name,
                       encoding="utf-8")
train_03_0 = train_03.iloc[0, :].copy()
train_03_0_df = pd.DataFrame(train_03_0).T
train_02_revised = pd.concat(objs=[train_02, train_03_0_df],
                             axis=0,
                             ignore_index=True)
train_02_revised_revised = train_02_revised.iloc[1:, :].copy()
train_02_revised_revised.to_csv("./train/train_02_revised.csv",
                                index=False)
a = pd.read_csv("./train/train_02_revised.csv",
                encoding="utf-8")

# 簡易的にtrain_01を訓練データとする
train_01 = pd.read_csv("./train/train_01_revised.csv",
                       encoding="utf-8")
train_01["id"] = train_01["id"].astype("int")
# 簡易的にtrain_02をテストデータとする
train_02 = pd.read_csv("./train/train_02_revised.csv",
                       encoding="utf-8")
train_02["id"] = train_02["id"].astype("int")

# 列毎にNullの個数を知りたいけれど、データ量が多すぎてinfo()では表示されない
train_01.info()

# for文を使って、列毎のNullの個数を確認
for column in columns_name:
  null_num = train_01[column].isnull().sum()
  print("{a}列でNullは{b}個".format(a=column, b=null_num))

# 平均、標準偏差、最小値、最大値、パーセンタイル値を確認
train_01.describe()

# 列が多すぎるので、目的変数を対象にdescribe()を確認
train_01["target"].describe()

# 列が多すぎるので、説明変数の最初の10列を対象にdescribe()を確認
ten_columns_name = columns_name[1:11]
train_01[ten_columns_name].describe()

# パーセンタイル値がピッタリ整数なので、説明変数の最初の10列を対象にして、データがどのような値かを確認
for column in ten_columns_name:
  print(train_01[column].value_counts())
