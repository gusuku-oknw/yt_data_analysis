import pandas as pd
import matplotlib.pyplot as plt
from transformers import pipeline
import torch
import re

# CSVファイルの読み込み
df = pd.read_csv('chat_messages_test.csv')


# データ前処理
def preprocess(df):
    # -1 を 0 に置き換え
    df = df[df['Time_in_seconds'] >= 0].copy()

    # 小数点以下を四捨五入して整数秒にする
    df['Time_in_seconds'] = df['Time_in_seconds'].round().astype(int)

    # スタンプや特殊文字を除去（同じ列に対する操作は一つの関数でまとめる）
    def clean_message(x):
        x = re.sub(r':_[A-Z]+:', '', str(x))
        x = re.sub(r'[^\w\s]', '', x)
        return x.strip()

    df['Message'] = df['Message'].apply(clean_message)

    return df


df = preprocess(df)

# GPUが利用可能かチェック
if torch.cuda.is_available():
    # GPUでのみ実行する感情分析パイプラインの設定
    sentiment_analyzer = pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english",
        tokenizer="distilbert-base-uncased",
        device=0  # GPUのデバイスID（通常は0）
    )
else:
    raise EnvironmentError("GPUが利用できません。この処理はGPU環境でのみ実行されます。")


# 感情分析をバッチ処理で実行
def analyze_sentiments(messages):
    results = sentiment_analyzer(list(messages), truncation=True, padding=True)
    labels = [result['label'] for result in results]
    scores = [round(float(result['score']), 5) for result in results]
    return labels, scores


# メッセージをバッチで処理し、結果をデータフレームに追加
batch_size = 64  # 適切なバッチサイズを設定
labels = []
scores = []

for i in range(0, len(df), batch_size):
    batch_messages = df['Message'].iloc[i:i + batch_size]
    batch_labels, batch_scores = analyze_sentiments(batch_messages)
    labels.extend(batch_labels)
    scores.extend(batch_scores)

df['Label'] = labels
df['Score'] = scores

# ポジティブなコメントのみを抽出
df['Positive'] = df['Label'] == 'POSITIVE'
positive_df = df[df['Positive']].copy()

# 10秒ごとに切り捨て（警告を回避するために `loc` を使用）
positive_df.loc[:, 'Time_in_10s'] = (positive_df['Time_in_seconds'] // 10) * 10

# 10秒ごとにポジティブコメント数を集計
ten_secondly_positive_counts = positive_df.groupby('Time_in_10s').size()

# 10秒ごとの時系列グラフの表示
plt.figure(figsize=(12, 6))
plt.plot(ten_secondly_positive_counts.index, ten_secondly_positive_counts.values, marker='o', linestyle='-',
         color='steelblue')
plt.xlabel('Time (seconds)')
plt.ylabel('Positive Comment Count')
plt.title('Positive Comment Counts per 10 Seconds')
plt.grid(True)
plt.tight_layout()
plt.show()

# 分析結果を新しいCSVファイルとして保存
df.to_csv('chat_messages_with_sentiment.csv', index=False, encoding='utf-8-sig')
