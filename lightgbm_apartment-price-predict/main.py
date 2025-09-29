from datetime import datetime
import re
import glob
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from xfeat import GBDTFeatureSelector
import optuna.integration.lightgbm as lgb_tune
from sklearn.metrics import mean_absolute_error
sns.set(font="Yu Gothic")


def eda(df):
    # 都道府県に注目してデータを調査(ヒストグラム)
    plt.figure(figsize=(30, 10))
    sns.histplot(data=df["都道府県名"])
    plt.title("都道府県毎のレコード数")
    plt.show()
    # 都道府県に注目してデータを調査(棒グラフ)
    ave_price_per_prefectures = df.groupby(["都道府県名"]).mean()["取引価格（総額）_log"].sort_values(ascending=False)
    plt.figure(figsize=(30, 10))
    sns.barplot(x=ave_price_per_prefectures.index, y=ave_price_per_prefectures.values)
    plt.ylim(bottom=6, top=8)
    plt.title("都道府県毎の取引価格の平均")
    plt.show()
    # 市区町村に注目してデータを調査(ヒストグラム)
    prefectures_list = df["都道府県名"].unique()
    for prefectures in prefectures_list:
        plt.figure(figsize=(30, 10))
        sns.histplot(data=df.query("都道府県名 == @prefectures")["市区町村名"])
        plt.xticks(rotation=90)
        plt.title(prefectures + "の市区町村毎のレコード数")
        plt.show()
    # 市区町村に注目してデータを調査(棒フラフ)
    prefectures_list = df["都道府県名"].unique()
    for prefectures in prefectures_list:
        ave_price_per_municipalities = df.query("都道府県名 == @prefectures").groupby(["市区町村名"]).mean()[
            "取引価格（総額）_log"].sort_values(ascending=False)
        plt.figure(figsize=(30, 10))
        sns.barplot(x=ave_price_per_municipalities.index, y=ave_price_per_municipalities.values)
        plt.xticks(rotation=90)
        plt.ylim(bottom=4, top=8)
        plt.title(prefectures + "の市区町村毎の平均取得価格")
        plt.show()
    # 間取りに注目してデータを調査(ヒストグラム)
    prefectures_list = df["都道府県名"].unique()
    for prefectures in prefectures_list:
        plt.figure(figsize=(30, 10))
        sns.histplot(
            data=df.query("都道府県名 == @prefectures")["間取り"].dropna())  # NaNはfloat扱いで、他の文字列とタイプが異なるので、NaNは弾く形にする
        plt.xticks(rotation=90)
        plt.title(prefectures + "の間取り毎のレコード数")
        plt.show()
    # 面積に注目してデータを調査(棒グラフ)
    prefectures_list = df["都道府県名"].unique()
    price_per_area_dict = {}
    for prefectures in prefectures_list:
        sum_area = df.query("都道府県名 == @prefectures")["面積（㎡）"].sum()
        sum_price = df.query("都道府県名 == @prefectures")["取引価格（総額）_log"].sum()
        price_per_area = sum_price / sum_area
        price_per_area_dict[prefectures] = price_per_area
    price_per_area_series = pd.Series(data=price_per_area_dict).sort_values(ascending=False)
    plt.figure(figsize=(30, 10))
    sns.barplot(x=price_per_area_series.index, y=price_per_area_series.values)
    plt.xticks(rotation=90)
    plt.xlabel("都道府県名")
    plt.ylabel("面積当たりの取引価格（総額）_log")
    plt.ylim(bottom=0.08, top=0.16)
    plt.show()
    # 最寄駅距離に注目してデータを調査(ヒストグラム)
    plt.figure(figsize=(30, 10))
    sns.histplot(data=df["最寄駅：距離（分）"])
    plt.show()
    # 最寄駅距離に注目してデータを調査(散布図)
    plt.figure(figsize=(30, 10))
    plt.scatter(x=df["最寄駅：距離（分）"].values, y=df["取引価格（総額）_log"].values)
    plt.show()
    # box-cox変換を用いての最寄駅距離に注目してデータを調査(散布図)
    df["最寄駅：距離（分）_boxcox"] = stats.boxcox(df["最寄駅：距離（分）"].values)[0]  # [1]を取得するとbox-cox変換のλ値になる
    plt.figure(figsize=(30, 10))
    plt.scatter(x=df["最寄駅：距離（分）_boxcox"].values, y=df["取引価格（総額）_log"].values)
    plt.show()
    # 特徴量に対しての相関係数(ヒートマップ)
    plt.figure(figsize=(30, 10))
    sns.heatmap(data=df.corr(), linewidths=1.0, annot=True)
    plt.show()


# 値の無い、もしくは値が1種類のみのカラムを削除
def drop_column(df):
    drop_column_list = []
    for k, v in zip(df.nunique().index, df.nunique().values):
        if v == 0 or v == 1:
            drop_column_list.append(k)
        else:
            continue
    df = df.drop(labels=drop_column_list, axis=1).copy()
    return df


# 値のある建築年の和暦を西暦に変換
def wareki_to_seireki(wareki):
    wareki = str(wareki)
    if wareki == "戦前":
        wareki = "昭和20年"
    if "元" in wareki:
        wareki.replace("元", "1")
    if "昭和" in wareki:
        value = wareki.split("昭和")[1].split("年")[0]
        seireki = 1925 + int(value)
    elif "平成" in wareki:
        value = wareki.split("平成")[1].split("年")[0]
        seireki = 1988 + int(value)
    elif "令和" in wareki:
        value = wareki.split("令和")[1].split("年")[0]
        seireki = 2018 + int(value)
    else:
        seireki = None
    return seireki


# 最寄駅：距離のフィールドをfloat型に変換
def moyori_kyori_to_float(df):
    distance_dict = {
        "30分?60分": "30",
        "1H?1H30": "30",
        "2H?": "30",
        "1H30?2H": "30"
    }
    df["最寄駅：距離（分）"] = df["最寄駅：距離（分）"].replace(to_replace=distance_dict).astype(float).copy()
    df["最寄駅：距離（分）"].fillna(value=15.0, inplace=True)
    return df


# 面積のフィールドをfloat型に変換
def menseki_to_float(df):
    area_square_meter_dict = {
        "2000㎡以上": "2000"
    }
    df["面積（㎡）"] = df["面積（㎡）"].replace(to_replace=area_square_meter_dict).astype(float).copy()
    return df


# 取引時点の文字部分を数字に変換
def torihiki_to_float(quarter):
    quarter_dict = {
        "年第1四半期": ".25",
        "年第2四半期": ".5",
        "年第3四半期": ".75",
        "年第4四半期": ".99"
    }
    for k, v in quarter_dict.items():
        quarter = str(quarter).replace(k, v)
    return quarter


# 駅名の価格情報を持っているDataFrameとmergeするためのキーとして、最寄駅：名称を編集
def adjust_station_name(station_name):
    if pd.isna(station_name):
        return "分からない"
    if "(" in station_name:
        return station_name.split("(")[0]
    else:
        return station_name


def data_preprocess(df, prefectures_data_df, kyori_df, construct_year_df):
    # 市区町村コードは市区町村名と同じ意味と推測されるので、市区町村コードの方のカラムを削除
    df.drop(labels=["市区町村コード"], axis=1, inplace=True)
    # 値のある建築年の和暦を西暦に変換
    df["建築年"] = df["建築年"].apply(wareki_to_seireki).copy()
    # 最寄駅：距離のフィールドをfloat型に変換
    df = moyori_kyori_to_float(df)
    # 面積のフィールドをfloat型に変換
    df = menseki_to_float(df)
    # 取引時点の文字部分を数字に変換
    df["取引時点"] = df["取引時点"].apply(torihiki_to_float).astype(float).copy()
    # 建築年から築年数を作成(スケーリング)
    df["築年数"] = df["建築年"].apply(lambda x: 2023-x).copy()
    df.drop(labels=["建築年"], axis=1, inplace=True)
    # 取引時点から取引経過年を作成(スケーリング)
    df["取引経過年"] = df["取引時点"].apply(lambda x: 2023.25-x).copy()
    df.drop(labels=["取引時点"], axis=1, inplace=True)
    # 改装、取引の事情等、間取り、都市計画、今後の利用目的、建物の構造、用途のNaNを分からないに変換(欠損値処理)
    nan_record_columns = ["改装", "取引の事情等", "間取り", "都市計画", "今後の利用目的", "建物の構造", "用途"]
    for nan_record_column in nan_record_columns:
        df[nan_record_column].fillna(value="分からない", inplace=True)
    # 都道府県毎の価格情報を持っているDataFrameとmergeして平均取得価格と面積当たり取得価格を取得(ビンカウンティング)
    # 最寄駅距離毎の価格情報を持っているDataFrameとmergeして価格変動率を取得(ビンカウンティング)
    # 築年数毎の価格情報を持っているDataFrameとmergeして価格変動率を取得(ビンカウンティング)
    df = df.merge(right=prefectures_data_df, how="left", on=["都道府県名"])
    df = df.merge(right=kyori_df, how="left", on=["最寄駅：距離（分）"])
    df = df.merge(right=construct_year_df, how="left", on=["築年数"])
    # 面積×面積当たり取得価格×距離毎価格変動率を格納(ペアワイズ交互作用特徴量)
    # 面積×面積当たり取得価格×年数毎価格変動率を格納(ペアワイズ交互作用特徴量)
    df["面積×都道府県面積当たり取得価格×距離毎価格変動率"] = df["面積（㎡）"] * df["都道府県毎の面積当たり取得価格"] * df["距離毎価格変動率"]
    df["面積×都道府県面積当たり取得価格×年数毎価格変動率"] = df["面積（㎡）"] * df["都道府県毎の面積当たり取得価格"] * df["年数毎価格変動率"]
    # Ｒ５価格_駅名のDataFrameとmergeするためのキーとなるカラムを作成した後にmergeしてＲ５価格_駅名を取得(外部データ利用)
    df["最寄駅：名称"] = df["最寄駅：名称"].apply(adjust_station_name).copy()
    df = df.merge(right=price_data_df, how="left", left_on=["最寄駅：名称"], right_on=["駅名"])
    # 面積×Ｒ５価格_駅名×距離毎価格変動率を格納(ペアワイズ交互作用特徴量)
    # 面積×Ｒ５価格_駅名×年数毎価格変動率を格納(ペアワイズ交互作用特徴量)
    df["面積×Ｒ５価格_駅名×距離毎価格変動率_log"] = np.log10(df["面積（㎡）"] * df["Ｒ５価格_駅名"] * df["距離毎価格変動率"])
    df["面積×Ｒ５価格_駅名×年数毎価格変動率_log"] = np.log10(df["面積（㎡）"] * df["Ｒ５価格_駅名"] * df["年数毎価格変動率"])
    df.drop(labels=["駅名", "Ｒ５価格_駅名"], axis=1, inplace=True)
    # Ｒ５価格_住居表示の価格情報を持っているDataFrameとmergeしてＲ５価格_住居表示を取得(外部データ利用)
    df = df.merge(right=price_data2_df, how="left", left_on=["地区名"], right_on=["住居表示"])
    # 面積×Ｒ５価格_住居表示×距離毎価格変動率を格納(ペアワイズ交互作用特徴量)
    # 面積×Ｒ５価格_住居表示×年数毎価格変動率を格納(ペアワイズ交互作用特徴量)
    df["面積×Ｒ５価格_住居表示×距離毎価格変動率_log"] = np.log10(df["面積（㎡）"] * df["Ｒ５価格_住居表示"] * df["距離毎価格変動率"])
    df["面積×Ｒ５価格_住居表示×年数毎価格変動率_log"] = np.log10(df["面積（㎡）"] * df["Ｒ５価格_住居表示"] * df["年数毎価格変動率"])
    df.drop(labels=["住居表示", "Ｒ５価格_住居表示"], axis=1, inplace=True)
    # 駅毎の乗降客数の情報を持っているDataFrameとmergeして駅毎乗降客数を取得(外部データ利用)
    df = df.merge(right=user_df, how="left", left_on=["最寄駅：名称"], right_on=["stationName"])
    df["駅毎乗降客数"].replace(to_replace={0.0: np.nan}, inplace=True)
    df["駅毎乗降客数_log"] = np.log10(df["駅毎乗降客数"])
    df.drop(labels=["stationName", "駅毎乗降客数"], axis=1, inplace=True)
    # 値の無い、もしくは値が1種類のみのカラムを削除
    df = drop_column(df)
    # 文字列変数をカテゴリ変数に変換
    for column_name in df.select_dtypes(include="object").columns:
        df[column_name] = df[column_name].astype("category").copy()
    return df


# データ前加工で使うDataFrameをcopy_train_dfから作成
def create_df_for_preprocess(copy_train_df):
    # copy_train_dfから都道府県毎の平均取得価格、面積当たり取得価格をDataFrameとして作成
    prefectures_data_df = copy_train_df.groupby(by=["都道府県名"]).mean(numeric_only=True)["取引価格（総額）_log"].reset_index()
    prefectures_data_df.rename(columns={"取引価格（総額）_log": "都道府県毎の平均取得価格"}, inplace=True)
    temp_prefectures_data_df = copy_train_df.groupby(by=["都道府県名"]).sum(numeric_only=True)[["面積（㎡）", "取引価格（総額）_log"]].reset_index()
    temp_prefectures_data_df.rename(columns={"面積（㎡）": "面積の合計", "取引価格（総額）_log": "取得価格の合計"}, inplace=True)
    prefectures_data_df = prefectures_data_df.merge(right=temp_prefectures_data_df, how="inner", on=["都道府県名"])
    prefectures_data_df["都道府県毎の面積当たり取得価格"] = prefectures_data_df["取得価格の合計"] / prefectures_data_df["面積の合計"]
    prefectures_data_df.drop(labels=["面積の合計", "取得価格の合計"], axis=1, inplace=True)
    # copy_train_dfから最寄駅距離毎の価格変動率をDataFrameとして作成
    kyori_df = copy_train_df.groupby(by=["最寄駅：距離（分）"]).mean(numeric_only=True)["取引価格（総額）_log"].reset_index()
    kyori_df.rename(columns={"取引価格（総額）_log": "距離毎の平均取得価格"}, inplace=True)
    kyori_df["距離毎価格変動率"] = kyori_df["距離毎の平均取得価格"] / kyori_df.loc[0, "距離毎の平均取得価格"]
    kyori_df.drop(labels=["距離毎の平均取得価格"], axis=1, inplace=True)
    # copy_train_dfから築年数毎の価格変動率をDataFrameとして作成
    construct_year_df = copy_train_df.groupby(by=["築年数"]).mean(numeric_only=True)["取引価格（総額）_log"].reset_index()
    construct_year_df.rename(columns={"取引価格（総額）_log": "年数毎の平均取得価格"}, inplace=True)
    construct_year_df["年数毎価格変動率"] = construct_year_df["年数毎の平均取得価格"] / construct_year_df.loc[0, "年数毎の平均取得価格"]
    construct_year_df.drop(labels=["年数毎の平均取得価格"], axis=1, inplace=True)
    return prefectures_data_df, kyori_df, construct_year_df


def main_select_feature(df):
    train_df, valid_df = train_test_split(df, test_size=0.2, random_state=42, shuffle=True)
    # データリーク対応としてcopy_train_dfを作成
    copy_train_df = train_df.copy()
    copy_train_df = menseki_to_float(copy_train_df)
    copy_train_df = moyori_kyori_to_float(copy_train_df)
    copy_train_df["建築年"] = copy_train_df["建築年"].apply(wareki_to_seireki).copy()
    copy_train_df["築年数"] = copy_train_df["建築年"].apply(lambda x: 2023 - x).copy()
    # データ前加工で使うDataFrameをcopy_train_dfから作成
    prefectures_data_df, kyori_df, construct_year_df = create_df_for_preprocess(copy_train_df)
    # データ前加工
    train_df = data_preprocess(train_df, prefectures_data_df, kyori_df, construct_year_df)
    valid_df = data_preprocess(valid_df, prefectures_data_df, kyori_df, construct_year_df)
    # 特徴量選択をxfeatで実施(パラメータの"metrics"は"metric"にする事)
    selector_param = {
        "objective": "regression",
        "metric": "mae"
    }
    selector_kwargs = {
        "num_boost_round": 1000,
    }
    selector = GBDTFeatureSelector(target_col="取引価格（総額）_log", threshold=0.75, lgbm_params=selector_param,
                                   lgbm_fit_kwargs=selector_kwargs)
    selector.fit_transform(train_df)
    print("選択された特徴量は", selector._selected_cols, sep=" ")


def main_cross_validation(df):
    drop_feature_column = ["ID", "都道府県名", "建ぺい率（％）", "取引価格（総額）_log"]
    # 交差検証(cross-validation)
    mae_value_list = []
    num_fold = 5  # fold値を5にしている時点で、trainは80%でvalidは20%となる
    kf = KFold(n_splits=num_fold, shuffle=True, random_state=42)
    print(datetime.now())
    for train, valid in kf.split(df):
        train_df = df.iloc[train]
        valid_df = df.iloc[valid]
        # データリーク対応としてcopy_train_dfを作成
        copy_train_df = train_df.copy()
        copy_train_df = menseki_to_float(copy_train_df)
        copy_train_df = moyori_kyori_to_float(copy_train_df)
        copy_train_df["建築年"] = copy_train_df["建築年"].apply(wareki_to_seireki).copy()
        copy_train_df["築年数"] = copy_train_df["建築年"].apply(lambda x: 2023 - x).copy()
        # データ前加工で使うDataFrameをcopy_train_dfから作成
        prefectures_data_df, kyori_df, construct_year_df = create_df_for_preprocess(copy_train_df)
        # データ前加工
        train_df = data_preprocess(train_df, prefectures_data_df, kyori_df, construct_year_df)
        valid_df = data_preprocess(valid_df, prefectures_data_df, kyori_df, construct_year_df)
        # 説明変数と目的変数に分けて、lightGBM用のデータセットを作成
        train_df_explained = train_df.drop(labels=drop_feature_column, axis=1)
        train_df_target = train_df["取引価格（総額）_log"].copy()
        valid_df_explained = valid_df.drop(labels=drop_feature_column, axis=1)
        valid_df_target = valid_df["取引価格（総額）_log"].copy()
        train_df_ds = lgb.Dataset(train_df_explained, train_df_target)
        valid_df_ds = lgb.Dataset(valid_df_explained, valid_df_target)
        # lightGBMモデルを作成
        param = {
            "objective": "regression",
            "metrics": "mae"
        }
        model = lgb.train(params=param, train_set=train_df_ds, valid_sets=valid_df_ds, num_boost_round=1000,
                          early_stopping_rounds=100, verbose_eval=-1)
        # MAEを算出してリストに格納
        valid_predict = model.predict(data=valid_df_explained)
        mae = mean_absolute_error(y_true=valid_df_target, y_pred=valid_predict)
        mae_value_list.append(mae)
    print(datetime.now())
    # MAEの平均を表示
    ave_mae = sum(mae_value_list) / len(mae_value_list)
    print(mae_value_list)
    print("MAEの平均は" + str(ave_mae))
    # 特徴量寄与度を算出
    df_feature_importance = pd.DataFrame(data={"特徴量寄与度": model.feature_importance()},
                                         index=train_df_explained.columns).sort_values(by=["特徴量寄与度"], ascending=False)
    print(df_feature_importance)
    print(train_df.columns)


def main_predict(df):
    drop_feature_column = ["ID", "都道府県名", "建ぺい率（％）", "取引価格（総額）_log"]
    train_df, valid_df = train_test_split(df, test_size=0.2, random_state=42, shuffle=True)
    # データリーク対応としてcopy_train_dfを作成
    copy_train_df = train_df.copy()
    copy_train_df = menseki_to_float(copy_train_df)
    copy_train_df = moyori_kyori_to_float(copy_train_df)
    copy_train_df["建築年"] = copy_train_df["建築年"].apply(wareki_to_seireki).copy()
    copy_train_df["築年数"] = copy_train_df["建築年"].apply(lambda x: 2023 - x).copy()
    # データ前加工で使うDataFrameをcopy_train_dfから作成
    prefectures_data_df, kyori_df, construct_year_df = create_df_for_preprocess(copy_train_df)
    # データ前加工
    print(train_df.shape, valid_df.shape, sep="\n")
    train_df = data_preprocess(train_df, prefectures_data_df, kyori_df, construct_year_df)
    valid_df = data_preprocess(valid_df, prefectures_data_df, kyori_df, construct_year_df)
    print(train_df.shape, valid_df.shape, sep="\n")
    # 説明変数と目的変数に分けて、lightGBM用のデータセットを作成
    train_df_explained = train_df.drop(labels=drop_feature_column, axis=1)
    train_df_target = train_df["取引価格（総額）_log"].copy()
    valid_df_explained = valid_df.drop(labels=drop_feature_column, axis=1)
    valid_df_target = valid_df["取引価格（総額）_log"].copy()
    train_df_ds = lgb.Dataset(train_df_explained, train_df_target)
    valid_df_ds = lgb.Dataset(valid_df_explained, valid_df_target)
    # lightGBMモデルを作成
    param = {
        "objective": "regression",
        "metrics": "mae"
    }
    model = lgb.train(params=param, train_set=train_df_ds, valid_sets=valid_df_ds, num_boost_round=1000,
                      early_stopping_rounds=100, verbose_eval=-1)
    # MAEを算出
    valid_predict = model.predict(data=valid_df_explained)
    mae = mean_absolute_error(y_true=valid_df_target, y_pred=valid_predict)
    print("MAEは" + str(mae))
    # 特徴量寄与度を算出
    df_feature_importance = pd.DataFrame(data={"特徴量寄与度": model.feature_importance()},
                                         index=train_df_explained.columns).sort_values(by=["特徴量寄与度"], ascending=False)
    print(df_feature_importance)
    # test_dfを予測
    test_drop_feature_column = ["ID", "都道府県名", "建ぺい率（％）"]
    test_df = pd.read_csv("./test.csv")
    print(test_df.shape)
    test_df = data_preprocess(test_df, prefectures_data_df, kyori_df, construct_year_df)
    print(test_df.shape)
    test_df_explained = test_df.drop(labels=test_drop_feature_column, axis=1)
    test_predict = model.predict(data=test_df_explained)
    test_df["取引価格（総額）_log"] = test_predict
    print(test_df.shape)
    test_df[["ID", "取引価格（総額）_log"]].to_csv("./test_submit.csv", index=False)


def main_predict_with_param_optimization(df):
    drop_feature_column = ["ID", "都道府県名", "建ぺい率（％）", "取引価格（総額）_log"]
    train_df, valid_df = train_test_split(df, test_size=0.2, random_state=42, shuffle=True)
    # データリーク対応としてcopy_train_dfを作成
    copy_train_df = train_df.copy()
    copy_train_df = menseki_to_float(copy_train_df)
    copy_train_df = moyori_kyori_to_float(copy_train_df)
    copy_train_df["建築年"] = copy_train_df["建築年"].apply(wareki_to_seireki).copy()
    copy_train_df["築年数"] = copy_train_df["建築年"].apply(lambda x: 2023 - x).copy()
    # データ前加工で使うDataFrameをcopy_train_dfから作成
    prefectures_data_df, kyori_df, construct_year_df = create_df_for_preprocess(copy_train_df)
    # データ前加工
    print(train_df.shape, valid_df.shape, sep="\n")
    train_df = data_preprocess(train_df, prefectures_data_df, kyori_df, construct_year_df)
    valid_df = data_preprocess(valid_df, prefectures_data_df, kyori_df, construct_year_df)
    print(train_df.shape, valid_df.shape, sep="\n")
    # 説明変数と目的変数に分けて、lightGBM用のデータセットを作成
    train_df_explained = train_df.drop(labels=drop_feature_column, axis=1)
    train_df_target = train_df["取引価格（総額）_log"].copy()
    valid_df_explained = valid_df.drop(labels=drop_feature_column, axis=1)
    valid_df_target = valid_df["取引価格（総額）_log"].copy()
    train_df_ds = lgb.Dataset(train_df_explained, train_df_target)
    valid_df_ds = lgb.Dataset(valid_df_explained, valid_df_target)
    # lightGBM TunerでlightGBMのパラメーターを最適化する場合、"metrics"でなく"metric"にする事
    param = {
        "objective": "regression",
        "metric": "mae"
    }
    # lightGBM Tuner(ベイズ最適化)でlightGBMの最適なパラメーターを取得
    print(datetime.now())
    model_param_tune = lgb_tune.train(params=param, train_set=train_df_ds, valid_sets=valid_df_ds)
    print(datetime.now())
    print(model_param_tune.params)
    # lightGBM Tuner(ベイズ最適化)で取得した最適なパラメータでlightGBMのモデルを作成
    model = lgb.train(params=model_param_tune.params, train_set=train_df_ds, valid_sets=valid_df_ds)
    # MAEを算出
    valid_predict = model.predict(data=valid_df_explained)
    mae = mean_absolute_error(y_true=valid_df_target, y_pred=valid_predict)
    print("MAEは" + str(mae))
    # 特徴量寄与度を算出
    df_feature_importance = pd.DataFrame(data={"特徴量寄与度": model.feature_importance()},
                                         index=train_df_explained.columns).sort_values(by=["特徴量寄与度"], ascending=False)
    print(df_feature_importance)
    # test_dfを予測
    test_drop_feature_column = ["ID", "都道府県名", "建ぺい率（％）"]
    test_df = pd.read_csv("./test.csv")
    print(test_df.shape)
    test_df = data_preprocess(test_df, prefectures_data_df, kyori_df, construct_year_df)
    print(test_df.shape)
    test_df_explained = test_df.drop(labels=test_drop_feature_column, axis=1)
    test_predict = model.predict(data=test_df_explained)
    test_df["取引価格（総額）_log"] = test_predict
    print(test_df.shape)
    test_df[["ID", "取引価格（総額）_log"]].to_csv("./test_submit.csv", index=False)


def main_predict_with_every_model_and_param_optimization(df):
    train_df, valid_df = train_test_split(df, test_size=0.2, random_state=42, shuffle=True)
    # データリーク対応としてcopy_train_dfを作成
    copy_train_df = train_df.copy()
    copy_train_df = menseki_to_float(copy_train_df)
    copy_train_df = moyori_kyori_to_float(copy_train_df)
    copy_train_df["建築年"] = copy_train_df["建築年"].apply(wareki_to_seireki).copy()
    copy_train_df["築年数"] = copy_train_df["建築年"].apply(lambda x: 2023 - x).copy()
    # データ前加工で使うDataFrameをcopy_train_dfから作成
    prefectures_data_df, kyori_df, construct_year_df = create_df_for_preprocess(copy_train_df)
    # trainとvalidデータ前加工
    print(train_df.shape, valid_df.shape, sep="\n")
    train_df = data_preprocess(train_df, prefectures_data_df, kyori_df, construct_year_df)
    valid_df = data_preprocess(valid_df, prefectures_data_df, kyori_df, construct_year_df)
    print(train_df.shape, valid_df.shape, sep="\n")
    # testデータ前加工
    test_df = pd.read_csv("./test.csv")
    test_df_prefecture_list = test_df["都道府県名"].unique()
    print(test_df.shape)
    test_df = data_preprocess(test_df, prefectures_data_df, kyori_df, construct_year_df)
    print(test_df.shape)
    # 都道府県名毎の予測モデル
    drop_feature_column = ["ID", "都道府県名", "建ぺい率（％）", "取引価格（総額）_log"]
    test_drop_feature_column = ["ID", "都道府県名", "建ぺい率（％）"]
    mae_dict = {}
    feature_importance_df_list = []
    test_prefecture_df_list = []
    print(datetime.now())
    for test_df_prefecture in test_df_prefecture_list:
        print("*****" + test_df_prefecture + "*****")
        # 説明変数と目的変数に分けて、lightGBM用のデータセットを作成
        training_df = train_df.query("都道府県名 == @test_df_prefecture").copy()
        validation_df = valid_df.query("都道府県名 == @test_df_prefecture").copy()
        training_df_explain = training_df.drop(labels=drop_feature_column, axis=1).copy()
        training_df_target = training_df["取引価格（総額）_log"].copy()
        validation_df_explain = validation_df.drop(labels=drop_feature_column, axis=1).copy()
        validation_df_target = validation_df["取引価格（総額）_log"].copy()
        training_ds = lgb.Dataset(training_df_explain, training_df_target)
        validation_ds = lgb.Dataset(validation_df_explain, validation_df_target)
        # lightGBM TunerでlightGBMのパラメーターを最適化する場合、"metrics"でなく"metric"にする事
        param = {
            "objective": "regression",
            "metric": "mae"
        }
        # lightGBM Tuner(ベイズ最適化)でlightGBMの最適なパラメーターを取得
        model_param_tune = lgb_tune.train(params=param, train_set=training_ds, valid_sets=validation_ds)
        # lightGBM Tuner(ベイズ最適化)で取得した最適なパラメータでlightGBMのモデルを作成
        model = lgb.train(params=model_param_tune.params, train_set=training_ds, valid_sets=validation_ds)
        # MAEを算出して格納
        validation_predict = model.predict(data=validation_df_explain)
        mae = mean_absolute_error(y_true=validation_df_target, y_pred=validation_predict)
        mae_dict[test_df_prefecture] = mae
        # 特徴量寄与度を算出して格納
        feature_importance_temp_df = pd.DataFrame(data=model.feature_importance(), columns=[test_df_prefecture],
                                                  index=training_df_explain.columns)
        feature_importance_df_list.append(feature_importance_temp_df)
        # 予測して格納
        test_prefecture_df = test_df.query("都道府県名 == @test_df_prefecture").copy()
        test_prefecture_df_explain = test_prefecture_df.drop(labels=test_drop_feature_column, axis=1).copy()
        test_prefecture_predict = model.predict(data=test_prefecture_df_explain)
        test_prefecture_df["取引価格（総額）_log"] = test_prefecture_predict
        test_prefecture_df_list.append(test_prefecture_df)
    # それぞれのMAEを表示
    print(mae_dict)
    # それぞれの特徴量寄与度を表示
    feature_importance_df = pd.concat(objs=feature_importance_df_list, axis=1, ignore_index=True)
    print(feature_importance_df)
    # それぞれの予測を纏めて、提出用の予測ファイルを作成
    test_submit_df = pd.concat(objs=test_prefecture_df_list, axis=0, ignore_index=True)
    test_submit_df.sort_values(by=["ID"], ascending=True, inplace=True)
    test_submit_df[["ID", "取引価格（総額）_log"]].to_csv("./test_submit2.csv", index=False)
    print(datetime.now())


def main_predict_with_every_model_and_nfold_and_param_optimization(df):
    test_df = pd.read_csv("./test.csv")
    test_df_prefecture_list = test_df["都道府県名"].unique()
    # データリーク対応としてcopy_train_dfを作成
    copy_train_df = df.copy()
    copy_train_df = menseki_to_float(copy_train_df)
    copy_train_df = moyori_kyori_to_float(copy_train_df)
    copy_train_df["建築年"] = copy_train_df["建築年"].apply(wareki_to_seireki).copy()
    copy_train_df["築年数"] = copy_train_df["建築年"].apply(lambda x: 2023 - x).copy()
    # データ前加工で使うDataFrameをcopy_train_dfから作成
    prefectures_data_df, kyori_df, construct_year_df = create_df_for_preprocess(copy_train_df)
    # trainとvalidデータ前加工
    df = data_preprocess(df, prefectures_data_df, kyori_df, construct_year_df)
    # testデータ前加工
    test_df = data_preprocess(test_df, prefectures_data_df, kyori_df, construct_year_df)
    test_df.sort_values(by=["ID"], ascending=True, inplace=True)
    # 都道府県名毎にNFold平均で予測モデル
    print(datetime.now())
    drop_feature_column = ["ID", "建ぺい率（％）", "取引価格（総額）_log"]
    test_drop_feature_column = ["ID", "建ぺい率（％）"]
    mae_dict = {}
    test_prefecture_df_list = []
    for test_df_prefecture in test_df_prefecture_list:
        print("*****" + test_df_prefecture + "*****")
        prefecture_df = df.query("都道府県名 == @test_df_prefecture").copy()
        prefecture_test_df = test_df.query("都道府県名 == @test_df_prefecture").copy()
        prefecture_test_df_explain = prefecture_test_df.drop(labels=test_drop_feature_column, axis=1)
        # NFold平均
        fold = 5
        kf = KFold(n_splits=fold, shuffle=True, random_state=42)
        prefecture_mae_list = []
        prefecture_test_predict_list = []
        for train, valid in kf.split(prefecture_df):
            prefecture_train_df = prefecture_df.iloc[train, :].copy()
            prefecture_valid_df = prefecture_df.iloc[valid, :].copy()
            # 説明変数と目的変数に分けて、lightGBM用のデータセットを作成
            training_df_explain = prefecture_train_df.drop(labels=drop_feature_column, axis=1).copy()
            training_df_target = prefecture_train_df["取引価格（総額）_log"].copy()
            validation_df_explain = prefecture_valid_df.drop(labels=drop_feature_column, axis=1).copy()
            validation_df_target = prefecture_valid_df["取引価格（総額）_log"].copy()
            training_ds = lgb.Dataset(training_df_explain, training_df_target)
            validation_ds = lgb.Dataset(validation_df_explain, validation_df_target)
            # lightGBM TunerでlightGBMのパラメーターを最適化する場合、"metrics"でなく"metric"にする事
            param = {
                "objective": "regression",
                "metric": "mae",
                "verbosity": -1
            }
            # lightGBM Tuner(ベイズ最適化)でlightGBMの最適なパラメーターを取得
            model_param_tune = lgb_tune.train(params=param, train_set=training_ds, valid_sets=validation_ds)
            # lightGBM Tuner(ベイズ最適化)で取得した最適なパラメータでlightGBMのモデルを作成
            model = lgb.train(params=model_param_tune.params, train_set=training_ds, valid_sets=validation_ds, verbose_eval=-1)
            # MAEを算出して格納
            validation_predict = model.predict(data=validation_df_explain)
            prefecture_mae = mean_absolute_error(y_true=validation_df_target, y_pred=validation_predict)
            prefecture_mae_list.append(prefecture_mae)
            prefecture_test_predict = model.predict(data=prefecture_test_df_explain)
            prefecture_test_predict_list.append(prefecture_test_predict)
        # MAEの平均を算出して格納
        mae = sum(prefecture_mae_list) / len(prefecture_mae_list)
        mae_dict[test_df_prefecture] = mae
        # 予測結果の平均を算出して、その平均を都道府県名毎の予測結果とする
        prefecture_test_predict_stack = np.stack(prefecture_test_predict_list, axis=1)  # 列方向に配列を追加する
        prefecture_test_predict_mean = np.mean(prefecture_test_predict_stack, axis=1)  # 行のカラムを1つにしたい場合はaxis=0、列のカラムを1つにしたい場合はaxis=1
        prefecture_test_df["取引価格（総額）_log"] = prefecture_test_predict_mean
        # 都道府県毎の予測したtestDataFrameをリストに格納
        test_prefecture_df_list.append(prefecture_test_df)
    # それぞれのMAEを表示
    print(mae_dict)
    # それぞれの予測を纏めて、提出用の予測ファイルを作成
    test_submit_df = pd.concat(objs=test_prefecture_df_list, axis=0, ignore_index=True)
    test_submit_df.sort_values(by=["ID"], ascending=True, inplace=True)
    test_submit_df[["ID", "取引価格（総額）_log"]].to_csv("./test_submit3.csv", index=False)
    print(datetime.now())


# trainやvalidやtestとmerge出来るように外部データのDataFrameの住居表示カラムを編集
def adjust_local_area_name(address):
    address = address.split("－")[0]
    if address:
        i = address[-1]
        if "０" in i or "１" in i or "２" in i or "３" in i or "４" in i or "５" in i or "６" in i or "７" in i or "８" in i or "９" in i:
            address = address[:-1]
            i = address[-1]
            if "０" in i or "１" in i or "２" in i or "３" in i or "４" in i or "５" in i or "６" in i or "７" in i or "８" in i or "９" in i:
                address = address[:-1]
    else:
        address = None
    return address


if __name__ == "__main__":
    files = glob.glob("./train/*.csv")
    df_list = []
    for file in files:
        df_list.append(pd.read_csv(file, converters={"面積（㎡）": str}))
    data_df = pd.concat(objs=df_list, axis=0, ignore_index=True)

    # 外部データ(参照元：https://nlftp.mlit.go.jp/ksj/old/datalist/old_KsjTmplt-L01.html)
    # 駅名とＲ５価格のDataFrame
    price_data_df = pd.read_csv("./L01-2023P-2K.csv")[["駅名", "Ｒ５価格"]]
    price_data_df.rename(columns={"Ｒ５価格": "Ｒ５価格_駅名"}, inplace=True)
    price_data_df = price_data_df.groupby(by=["駅名"]).mean().reset_index()
    # 住居表示とＲ５価格のDataFrame
    price_data2_df = pd.read_csv("./L01-2023P-2K.csv")[["住居表示", "Ｒ５価格"]]
    price_data2_df.rename(columns={"Ｒ５価格": "Ｒ５価格_住居表示"}, inplace=True)
    price_data2_df["住居表示"] = price_data2_df["住居表示"].apply(adjust_local_area_name).copy()
    price_data2_df = price_data2_df.groupby(by=["住居表示"]).mean().reset_index()
    # 外部データ(参照元：https://nlftp.mlit.go.jp/ksj/gml/datalist/KsjTmplt-S12-v3_1.html)
    # 駅名と乗降客数のDataFrame
    temp_user_df = pd.read_xml("./S12-22.xml")[["stationName", "passengers2021"]]
    temp_user_df.dropna(axis=0, inplace=True)
    temp_user_df.rename(columns={"passengers2021": "駅毎乗降客数"}, inplace=True)
    user_df = temp_user_df.groupby(by=["stationName"]).max(numeric_only=True)["駅毎乗降客数"].reset_index()

    # eda(data_df)
    # main_select_feature(data_df)
    # main_cross_validation(data_df)
    # main_predict(data_df)
    # main_predict_with_param_optimization(data_df)
    # main_predict_with_every_model_and_param_optimization(data_df)
    main_predict_with_every_model_and_nfold_and_param_optimization(data_df)
