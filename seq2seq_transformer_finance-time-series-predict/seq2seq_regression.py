from datetime import datetime
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
torch.manual_seed(3407)


# Attention付きSeq2Seqモデルのクラスを作成
class Seq2SeqAttention(nn.Module):
    def __init__(self, in_n, hidden_n, out_n):
        super().__init__()
        self.layer_1_encoder = nn.Linear(in_n, hidden_n)
        self.layer_2_encoder = nn.LSTM(hidden_n, hidden_n, batch_first=True)
        self.layer_1_decoder = nn.Linear(in_n, hidden_n)
        self.layer_2_decoder = nn.LSTM(hidden_n, hidden_n, batch_first=True)
        self.layer_3_decoder = nn.Linear(hidden_n*2, hidden_n)
        self.layer_4_decoder = nn.Linear(hidden_n, hidden_n//2)
        self.dropout_decoder = nn.Dropout(0.25)
        self.layer_5_decoder = nn.Linear(hidden_n//2, out_n)
    
    def forward(self, input_encoder, input_decoder):
        out_layer_1_encoder = F.tanh(self.layer_1_encoder(input_encoder))  # tanhを活性化関数
        out_layer_2_encoder, (hn_enc, cn_enc) = self.layer_2_encoder(out_layer_1_encoder)
        out_layer_1_decoder = F.tanh(self.layer_1_decoder(input_decoder))  # tanhを活性化関数
        out_layer_2_decoder, (hn_dec, cn_dec) = self.layer_2_decoder(out_layer_1_decoder, (hn_enc, cn_enc))
        out_layer_2_encoder_matrix_swap = out_layer_2_encoder.permute(0, 2, 1)
        out_lstm_decoder_dot_encoder = torch.bmm(out_layer_2_decoder, out_layer_2_encoder_matrix_swap)
        minibatch_size, decoder_time_series, encoder_time_series = out_lstm_decoder_dot_encoder.shape
        out_lstm_decoder_dot_encoder_matrix = out_lstm_decoder_dot_encoder.reshape(minibatch_size*decoder_time_series, encoder_time_series)
        similarity_matrix = F.softmax(out_lstm_decoder_dot_encoder_matrix, dim=1)
        similarity_alpha = similarity_matrix.reshape(minibatch_size, decoder_time_series, encoder_time_series)
        context_vector = torch.bmm(similarity_alpha, out_layer_2_encoder)
        out_layer_2_decoder_attention = torch.cat([context_vector, out_layer_2_decoder], dim=2)
        out_layer_3_decoder = F.tanh(self.layer_3_decoder(out_layer_2_decoder_attention))  # tanhを活性化関数
        out_layer_4_decoder = F.tanh(self.layer_4_decoder(out_layer_3_decoder))  # tanhを活性化関数
        out_dropout_decoder = self.dropout_decoder(out_layer_4_decoder)
        output_decoder = self.layer_5_decoder(out_dropout_decoder)
        output_decoder = self.layer_5_decoder(out_layer_4_decoder)
        return output_decoder


# datasetを作るためのクラス関数
class MyDataset(Dataset):
    def __init__(self, chunk, df, drop_labels):
        self._chunk = chunk
        encoder_data_rows = 1 * self._chunk
        decoder_data_rows = 2 * self._chunk
        encoder_index_from = 0
        encoder_index_to = encoder_data_rows
        decoder_index_from = encoder_data_rows
        decoder_index_to = encoder_data_rows + decoder_data_rows
        self._explain_encoder = []
        self._explain_decoder = []
        self._target_decoder = []
        while True:
            if decoder_index_to > df.shape[0]:
                break
            else:
                explain_encoder_df = df.iloc[encoder_index_from:encoder_index_to, :].drop(labels=drop_labels, axis=1).copy()
                self._explain_encoder.append(explain_encoder_df)  # [dataframe(10000, 697), dataframe(10000, 697), dataframe(10000, 697), .....]の形
                explain_decoder_df = df.iloc[decoder_index_from:decoder_index_to, :].drop(labels=drop_labels, axis=1).copy()
                self._explain_decoder.append(explain_decoder_df)  # [dataframe(20000, 697), dataframe(20000, 697), dataframe(20000, 697), .....]の形
                target_decoder_df = df[["target"]].iloc[decoder_index_from:decoder_index_to, :].copy()
                self._target_decoder.append(target_decoder_df)
                encoder_index_from += self._chunk
                encoder_index_to += self._chunk
                decoder_index_from += self._chunk
                decoder_index_to += self._chunk
    
    def __len__(self):
        length_target_decoder = len(self._target_decoder)
        return length_target_decoder  # このreturnの値が__getitem__のidxの値になる
    
    def __getitem__(self, idx):  # 今回はidxの値は64になり、イテレータとしてidx=0,1,2,,,,,63で64回forループする
        # -> ここの処理はfor hoge, waku, teka in dataset_minibatches:のforループの所で行われる
        _explain_encoder_array = np.array(self._explain_encoder[idx])
        _explain_encoder_tensor = torch.FloatTensor(_explain_encoder_array)  # それぞれは(10000, 695)のtensorの形
        _explain_decoder_array = np.array(self._explain_decoder[idx])
        _explain_decoder_tensor = torch.FloatTensor(_explain_decoder_array)  # それぞれは(20000, 695)のtensorの形
        _target_decoder_array = np.array(self._target_decoder[idx])
        _target_decoder_tensor = torch.FloatTensor(_target_decoder_array)  # それぞれは(10000, 1)のtensorの形
        return _explain_encoder_tensor, _explain_decoder_tensor, _target_decoder_tensor


# datasetをミニバッチ化する関数(DataLoader関数での引数)
# -> ここの処理はfor hoge, waku, teka in dataset_minibatches:のforループの所で行われる
# -> ミニバッチ単位に処理が繰り返される形であり、今回のdatasetはそれぞれ64個のtensorで、3ミニバッチにするので、
#    22回処理が繰り返されて、21個の(tensor, tensor, tensor)と、1個の(tensor,)が作られる形になる
def my_collate_fn(batch):
    explain_encoder, explain_decoder, target_decoder = list(zip(*batch))  # (tensor, tensor, tensor)の形　※厳密には(tensor, tensor, tensor)が21回、(tensor,)が1回作られる
    explain_encoder_minibatch = list(explain_encoder)  # [tensor, tensor, tensor]の形　※厳密には[tensor, tensor, tensor]が21回、[tensor]が1回作られる
    explain_decoder_minibatch = list(explain_decoder)
    target_decoder_minibatch = list(target_decoder)
    return explain_encoder_minibatch, explain_decoder_minibatch, target_decoder_minibatch


# 教師データやテストデータのDataFrameを1ミニバッチTensorに変換する関数
def df_2_tensor_onebatch(df, drop_labels):
    df_drop = df.drop(labels=drop_labels, axis=1).copy()
    df_array = np.array(df_drop)
    df_tensor = torch.FloatTensor(df_array)
    df_tensor_onebatch = df_tensor.unsqueeze(0).to(device)  # GPUへ
    return df_tensor_onebatch


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
# Attention付きSeq2Seqモデルの引数となるパーセプトロン数を設定
input_number = 695
hidden_number = 100
output_number = 1
# Attention付きSeq2Seqモデルのインスタンスを作成、最適化関数および損失関数を設定
nn_model = Seq2SeqAttention(in_n=input_number,
                            hidden_n=hidden_number,
                            out_n=output_number).to(device)  # GPUへ
optimizer = optim.Adam(nn_model.parameters(),
                       lr=0.01)
loss_f = nn.MSELoss()
# 学習
nn_model.train()
chunk_number = 10000
minibatch_size = 3
train_df = train.copy()
if os.path.isfile("./FinanceTimeSeries.model"):
    nn_model.load_state_dict(torch.load("./FinanceTimeSeries.model"))
scaler = torch.cuda.amp.GradScaler(init_scale=4096)
for times in range(10):
    kanryou = times * 10
    print("{a}エポック完了".format(a=kanryou))
    for epoch in range(10):
        print("******************************")
        print("{a}エポック目開始".format(a=epoch+1))
        print(datetime.now())
        epoch_sum_loss = 0
        # 訓練データをdatasetの形にして、ミニバッチ化
        datasets = MyDataset(chunk=chunk_number,
                             df=train_df,
                             drop_labels=["id", "target"])
        dataset_minibatches = DataLoader(datasets,
                                         batch_size=minibatch_size,
                                         num_workers=os.cpu_count(),
                                         shuffle=False,
                                         collate_fn=my_collate_fn)
        for train_explain_encoder_minibatch, train_explain_decoder_minibatch, train_target_decoder_minibatch in dataset_minibatches:
            train_explain_encoder_minibatch_tensor = torch.stack(train_explain_encoder_minibatch,
                                                                 dim=2)  # (10000, 695, 3)のtensorの形
            train_explain_encoder_minibatch_tensor = train_explain_encoder_minibatch_tensor.permute(2, 0, 1).to(device)  # ※厳密には(3, 10000, 695)が21回、(1, 10000, 695)が1回の形
            train_explain_decoder_minibatch_tensor = torch.stack(train_explain_decoder_minibatch,
                                                                 dim=2)
            train_explain_decoder_minibatch_tensor = train_explain_decoder_minibatch_tensor.permute(2, 0, 1).to(device)
            train_target_decoder_minibatch_tensor = torch.stack(train_target_decoder_minibatch,
                                                                dim=2)
            train_target_decoder_minibatch_tensor = train_target_decoder_minibatch_tensor.permute(2, 0, 1).to(device)
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                output = nn_model(input_encoder=train_explain_encoder_minibatch_tensor,
                                  input_decoder=train_explain_decoder_minibatch_tensor)
                loss = 0
                for i in range(output.shape[0]):
                    loss += loss_f(output[i],
                                   train_target_decoder_minibatch_tensor[i])
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
chunk_number = 10000
decoder_data_rows = 2 * chunk_number
train_df = train.copy()
test_df = test.copy()
with torch.no_grad():
    nn_model.load_state_dict(torch.load("./FinanceTimeSeries.model"))
    # encoder、decoder向けにテストデータを前処理
    test_encoder = train_df.iloc[-1*chunk_number:, :].copy()
    test_explain_encoder_tensor_onebatch = df_2_tensor_onebatch(df=test_encoder,
                                                                drop_labels=["id", "target"])
    decoder_index_from = 0
    decoder_index_to = decoder_data_rows
    output_predict_df_list = []
    while True:
        if decoder_index_to < test_df.shape[0]:
            test_decoder = test_df.iloc[decoder_index_from:decoder_index_to, :].copy()
            test_explain_decoder_tensor_onebatch = df_2_tensor_onebatch(df=test_decoder,
                                                                        drop_labels=["id"])
        elif decoder_index_to == 280000:
            test_decoder = test_df.iloc[decoder_index_from:, :].copy()
            test_explain_decoder_tensor_onebatch = df_2_tensor_onebatch(df=test_decoder,
                                                                        drop_labels=["id"])
        else:
            break
        # 予測(dropout層を用いない形で推論する)
        output_layer_1_encoder = F.tanh(nn_model.layer_1_encoder(test_explain_encoder_tensor_onebatch))  # tanhを活性化関数
        output_layer_2_encoder, (hn, cn) = nn_model.layer_2_encoder(output_layer_1_encoder)
        output_layer_1_decoder = F.tanh(nn_model.layer_1_decoder(test_explain_decoder_tensor_onebatch))  # tanhを活性化関数
        output_layer_2_decoder, (hn, cn) = nn_model.layer_2_decoder(output_layer_1_decoder, (hn, cn))
        output_layer_2_encoder_matrix_swap = output_layer_2_encoder.permute(0, 2, 1)
        output_lstm_decoder_dot_encoder = torch.bmm(output_layer_2_decoder, output_layer_2_encoder_matrix_swap)
        minibatch_size, decoder_time_series, encoder_time_series = output_lstm_decoder_dot_encoder.shape
        output_lstm_decoder_dot_encoder_matrix = output_lstm_decoder_dot_encoder.reshape(minibatch_size*decoder_time_series, encoder_time_series)
        similarity_matrix = F.softmax(output_lstm_decoder_dot_encoder_matrix, dim=1)
        similarity_alpha = similarity_matrix.reshape(minibatch_size, decoder_time_series, encoder_time_series)
        context_vector = torch.bmm(similarity_alpha, output_layer_2_encoder)
        output_layer_2_decoder_attention = torch.cat([context_vector, output_layer_2_decoder], dim=2)
        output_layer_3_decoder = F.tanh(nn_model.layer_3_decoder(output_layer_2_decoder_attention))  # tanhを活性化関数
        output_layer_4_decoder = F.tanh(nn_model.layer_4_decoder(output_layer_3_decoder))  # tanhを活性化関数
        output = nn_model.layer_5_decoder(output_layer_4_decoder)
        output_predict = np.array(output[0].to("cpu"))
        test_decoder_id = np.array(test_decoder[["id"]].copy())
        output_predict_df = pd.DataFrame(data={"id": test_decoder_id.T[0],
                                               "predict_target": output_predict.T[0]})
        output_predict_df_list.append(output_predict_df)
        decoder_index_from += decoder_data_rows
        decoder_index_to += decoder_data_rows
    submit = pd.concat(objs=output_predict_df_list,
                       axis=0,
                       ignore_index=True)
