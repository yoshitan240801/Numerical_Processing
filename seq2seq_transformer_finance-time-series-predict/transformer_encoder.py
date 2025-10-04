from datetime import datetime
import math
import os

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
torch.manual_seed(3407)


# 位置エンコーディングのクラスを作成(参考：https://qiita.com/age884/items/bc27c532a6d7c720fc3e)
class PositionalEncoding(nn.Module):
    def __init__(self, hidden_n, max_time_series, dropout_rate=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout_rate)

        position_encode = torch.zeros(max_time_series, hidden_n)  # (最大時系列数, transformer層パーセプトロン数)のゼロ行列を作成

        position = torch.arange(0, max_time_series, dtype=torch.float)  # [0, 1, 2, ,,,]の最大時系列数の行ベクトルを作成
        position = position.unsqueeze(1)  # 最大時系列数の行ベクトルを(最大時系列数, 1)の行列に変換

        division_term = torch.arange(0, hidden_n, 2, dtype=torch.float)  # [0, 2, 4, ,,,]のtransformer層パーセプトロン数の半分の値の行ベクトルを作成
        division_term = torch.exp(division_term * (-math.log(10000.0) / hidden_n))  # 行ベクトルの各値に対して所定の計算

        position_encode_even_columns = torch.sin(position * division_term)  # (最大時系列数, transformer層パーセプトロン数/2)の行列
        position_encode[:, 0::2] = position_encode_even_columns  # position_encode行列の全ての行の偶数列に代入
        position_encode_odd_columns = torch.cos(position * division_term)
        position_encode[:, 1::2] = position_encode_odd_columns  # position_encode行列の全ての行の奇数列に代入

        position_encode = position_encode.unsqueeze(0)  # (1, 最大時系列数, transformer層パーセプトロン数)のテンソルに変換

        self.register_buffer("pe", position_encode)

    def forward(self, input):
        input = input + self.pe[:, :input.shape[1], :]  # inputのミニバッチのそれぞれにposition_encodeを加算
        return self.dropout(input)


# Transformerモデルのクラスを作成　※encoderのみ
class MyTransformer(nn.Module):  # num_headが8だとGPUでOutOfMemoryになる
    def __init__(self, in_n, hidden_n, out_n, time_series, num_transformer_layer_encoder,
                 num_head=5, dimension_feedforward=512, dropout_rate=0.2):
        super().__init__()
        self.layer_1_encoder = nn.Linear(in_n, hidden_n)
        self.layer_2_encoder = PositionalEncoding(hidden_n=hidden_n, max_time_series=time_series)
        transformer_layer_encoder = nn.TransformerEncoderLayer(d_model=hidden_n,
                                                               nhead=num_head,
                                                               dim_feedforward=dimension_feedforward,
                                                               dropout=dropout_rate,
                                                               activation="gelu",
                                                               layer_norm_eps=1e-5,
                                                               batch_first=True,
                                                               norm_first=False)
        layer_normalization_encoder = nn.LayerNorm(hidden_n)
        self.layer_3_encoder = nn.TransformerEncoder(encoder_layer=transformer_layer_encoder,
                                                     num_layers=num_transformer_layer_encoder,
                                                     norm=layer_normalization_encoder,
                                                     enable_nested_tensor=True,
                                                     mask_check=True)
        self.layer_4_encoder = nn.Linear(hidden_n, out_n)
        print(self)

    def forward(self, input_encoder):
        out_layer_1_encoder = self.layer_1_encoder(input_encoder)
        out_layer_2_encoder = self.layer_2_encoder(input=out_layer_1_encoder)
        out_layer_3_encoder = self.layer_3_encoder(src=out_layer_2_encoder,
                                                   mask=None,
                                                   is_causal=None,
                                                   src_key_padding_mask=None)
        output_encoder = self.layer_4_encoder(out_layer_3_encoder)
        return output_encoder


# datasetを作るためのクラス関数
class MyDataset(Dataset):
    def __init__(self, chunk, df, explain_feature):
        encoder_data_rows = 1 * chunk  # 時系列データ数は5万だとGPUでOutOfMemoryになる
        encoder_index_from = 0
        encoder_index_to = encoder_data_rows
        self._explain_encoder = []
        self._target_encoder = []
        while True:
            if encoder_index_to > df.shape[0]:
                break
            else:
                explain_encoder_df = df[explain_feature].iloc[encoder_index_from:encoder_index_to, :].copy()
                self._explain_encoder.append(explain_encoder_df)
                target_encoder_df = df[["target_class"]].iloc[encoder_index_from:encoder_index_to, :].copy()
                self._target_encoder.append(target_encoder_df)
                encoder_index_from += chunk
                encoder_index_to += chunk

    def __len__(self):
        length_target_encoder = len(self._target_encoder)
        return length_target_encoder

    def __getitem__(self, idx):
        _explain_encoder_array = np.array(self._explain_encoder[idx])
        _explain_encoder_tensor = torch.FloatTensor(_explain_encoder_array)
        _target_encoder_array = np.array(self._target_encoder[idx])
        _target_encoder_tensor = torch.LongTensor(_target_encoder_array)
        return _explain_encoder_tensor, _target_encoder_tensor


# datasetをミニバッチ化する関数(DataLoader関数での引数)
def my_collate_fn(batch):
    explain_encoder, target_encoder = list(zip(*batch))
    explain_encoder_minibatch = list(explain_encoder)
    target_encoder_minibatch = list(target_encoder)
    return explain_encoder_minibatch, target_encoder_minibatch


# DataFrameを1ミニバッチTensorに変換する関数
def df_2_tensor_onebatch(df, explain_feature):
    df_copy = df[explain_feature].copy()
    df_array = np.array(df_copy)
    df_tensor = torch.FloatTensor(df_array)
    df_tensor_onebatch = df_tensor.unsqueeze(0)
    layer_normalization = nn.LayerNorm(df_tensor_onebatch.shape[2])
    df_tensor_onebatch_ln = layer_normalization(df_tensor_onebatch).to(device)  # GPUへ
    return df_tensor_onebatch_ln


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# 訓練データをcsvから軽量化してparquetへ
train = pd.read_csv("./train.csv",
                    index_col=["id", "target"],  # idとtargetをインデックスとして読み込み(カラムにしない事で型変換に巻き込ませない)
                    encoding="utf-8")
train = train.astype(np.int8)  # カラムの型をint8へ(idとtargetはインデックスなので型変換されない)
train = train.reset_index(level=["id", "target"])  # idとtargetをインデックスからカラムへ
train.to_parquet("./train.parquet")
# テストデータをcsvから軽量化してparquetへ
test = pd.read_csv("./test.csv",
                   index_col=["id"],  # idをインデックスとして読み込み(カラムにしない事で型変換に巻き込ませない)
                   encoding="utf-8")
test = test.astype(np.int8)  # カラムの型をint8へ(idはインデックスなので型変換されない)
test = test.reset_index(level=["id"])  # idをインデックスからカラムへ
test.to_parquet("./test.parquet")
# 訓練データを読み込み(read_parquetにencodingの引数は無い)
train = pd.read_parquet("./train.parquet")
# テストデータを読み込み(read_parquetにencodingの引数は無い)
test = pd.read_parquet("./test.parquet")
# 訓練データの目的変数を使って、目的変数の値を分類分けする辞書を作成
value2id_dict = {}
id2value_dict = {}
num = 0
for i, v in enumerate(train["target"].unique()):
    value2id_dict[v] = i
    id2value_dict[i] = v
# 特徴量選択のためLightGBMを実施
train_df_lgb = train.copy()
train_df_lgb["target_class"] = train_df_lgb["target"].apply(lambda x: value2id_dict[x]).copy()
training_df, validation_df = train_test_split(train_df_lgb,
                                              test_size=0.2,
                                              random_state=42,
                                              shuffle=True)
training_explain_df = training_df.drop(labels=["id", "target", "target_class"],
                                       axis=1).copy()
training_target_df = training_df[["target_class"]].copy()
validation_explain_df = validation_df.drop(labels=["id", "target", "target_class"],
                                           axis=1).copy()
validation_target_df = validation_df[["target_class"]].copy()
training_ds = lgb.Dataset(training_explain_df,
                          training_target_df)
validation_ds = lgb.Dataset(validation_explain_df,
                            validation_target_df)
param = {
    "objective": "multiclass",
    "num_class": 10,
    "metrics": ["auc_mu", "multi_logloss"],
    "random_state": 42,
    "verbose": -1
}
model = lgb.train(params=param,
                  train_set=training_ds,
                  valid_sets=validation_ds,
                  num_boost_round=1000,
                  callbacks=[lgb.early_stopping(stopping_rounds=100,
                                                verbose=True)])
feature_importance_df = pd.DataFrame(data={"特徴量寄与度": model.feature_importance()},
                                     index=training_explain_df.columns).sort_values(by=["特徴量寄与度"],
                                                                                    ascending=False)
feature_importance_df.to_pickle("./feature_importance_df.pkl")
feature_importance_df = pd.read_pickle("./feature_importance_df.pkl")
explain_feature_list = list(feature_importance_df.query("特徴量寄与度 >= 500").index)
# Transformerモデルの引数となるパーセプトロン数を設定
input_number = len(explain_feature_list)
hidden_number = 180  # num_headの倍数でないとエラーになる＆偶数の値でないとPositionalEncodingの[:, 0::2]と[:, 1::2]でエラーになる
output_number = len(train["target"].unique())
chunk_number = 10000
max_time_series_number = 1 * chunk_number  # 時系列データ数は5万だとGPUでOutOfMemoryになる
number_transformer_layer_encoder = 2
# Transformerモデルのインスタンスを作成、最適化関数および損失関数を設定
nn_model = MyTransformer(in_n=input_number,
                         hidden_n=hidden_number,
                         time_series=max_time_series_number,
                         num_transformer_layer_encoder=number_transformer_layer_encoder,
                         out_n=output_number).to(device)  # GPUへ
optimizer = optim.Adam(nn_model.parameters(),
                       lr=0.001)
loss_f = nn.CrossEntropyLoss()
# 学習
nn_model.train()
minibatch_size = 1
train_df = train.copy()
train_df["target_class"] = train_df["target"].apply(lambda x:value2id_dict[x]).copy()
if os.path.isfile("./FinanceTimeSeries.model"):
    nn_model.load_state_dict(torch.load("./FinanceTimeSeries.model"))
scaler = torch.cuda.amp.GradScaler(init_scale=4096)
for epoch in range(50):
    print("******************************")
    print("{a}エポック目開始".format(a=epoch+1))
    print(datetime.now())
    epoch_sum_loss = 0
    # 訓練データをdatasetの形にして、ミニバッチ化
    datasets = MyDataset(chunk=chunk_number,
                         df=train_df,
                         explain_feature=explain_feature_list)
    dataset_minibatches = DataLoader(datasets,
                                     batch_size=minibatch_size,
                                     shuffle=False,
                                     collate_fn=my_collate_fn)
    for train_explain_encoder_minibatch, train_target_encoder_minibatch in dataset_minibatches:
        train_explain_encoder_minibatch_tensor = torch.stack(train_explain_encoder_minibatch,
                                                             dim=0)
        layer_normalization_explain_encoder = nn.LayerNorm(train_explain_encoder_minibatch_tensor.shape[2])
        train_explain_encoder_minibatch_tensor_after_ln = layer_normalization_explain_encoder(train_explain_encoder_minibatch_tensor).to(device)
        train_target_encoder_minibatch_tensor = torch.stack(train_target_encoder_minibatch,
                                                            dim=0).to(device)
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            output = nn_model(input_encoder=train_explain_encoder_minibatch_tensor_after_ln)
            loss = 0
            for i in range(output.shape[0]):
                loss += loss_f(output[i], train_target_encoder_minibatch_tensor[i].T[0])  # 正解データは[i]でミニバッチ毎に行列にして、Tで行列を転置した後、[0]でベクトルとする
        optimizer.zero_grad(set_to_none=True)
        # loss.backward()
        scaler.scale(loss).backward()  # autocast利用時(逆伝播)
        # optimizer.step()
        scaler.unscale_(optimizer)  # autocast利用時(勾配クリッピングの前にスケールを戻す)
        nn.utils.clip_grad_norm_(nn_model.parameters(),
                                 max_norm=0.8)  # autocast利用時(勾配クリッピング)
        scaler.step(optimizer)  # autocast利用時(勾配を更新)
        scaler.update()  # autocast利用時(スケールを更新)
        epoch_sum_loss += loss.item()
    print(epoch_sum_loss)
    torch.save(nn_model.state_dict(),
               "./FinanceTimeSeries.model")
print(datetime.now())
# 推論
nn_model.eval()
encoder_data_rows = 1 * chunk_number
test_df = test.copy()
with torch.no_grad():
    nn_model.load_state_dict(torch.load("./FinanceTimeSeries.model"))
    # encoder向けにテストデータを前処理
    encoder_index_from = 0
    encoder_index_to = encoder_data_rows
    output_predict_df_list = []
    while True:
        if encoder_index_to < test_df.shape[0]:
            test_encoder = test_df.iloc[encoder_index_from:encoder_index_to, :].copy()
            test_explain_encoder_tensor_onebatch_after_ln = df_2_tensor_onebatch(df=test_encoder,
                                                                                 explain_feature=explain_feature_list)
        elif encoder_index_to == 270000:
            test_encoder = test_df.iloc[encoder_index_from:, :].copy()
            test_explain_encoder_tensor_onebatch_after_ln = df_2_tensor_onebatch(df=test_encoder,
                                                                                 explain_feature=explain_feature_list)
        else:
            break
        # 予測(nn_model.eval()によってクラスの方でイイ感じにdropout層を用いない形で推論する)
        output = nn_model(input_encoder=test_explain_encoder_tensor_onebatch_after_ln)
        output_argmax_after_softmax = torch.argmax(input=F.softmax(output[0], dim=1), dim=1)
        output_predict = np.array(output_argmax_after_softmax.to("cpu"))
        test_encoder_id = np.array(test_encoder[["id"]].copy())
        output_predict_df = pd.DataFrame(data={"id": test_encoder_id.T[0],
                                               "predict_target_class": output_predict})
        output_predict_df["predict_target"] = output_predict_df["predict_target_class"].apply(lambda x:id2value_dict[x]).copy()
        output_predict_df_list.append(output_predict_df)
        encoder_index_from += encoder_data_rows
        encoder_index_to += encoder_data_rows
    submit = pd.concat(objs=output_predict_df_list,
                       axis=0,
                       ignore_index=True)
