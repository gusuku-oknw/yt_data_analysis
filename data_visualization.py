import pandas as pd
import matplotlib.pyplot as plt
from transformers import pipeline
import torch
import re
import tqdm

# CSVファイルの読み込み
df = pd.read_csv('chat_messages_test.csv')


# データ前処理
def preprocess(df):
    # -1 を 0 に置き換え
    df = df[df['Time_in_seconds'] >= 0].copy()

    # 小数点以下を四捨五入して整数秒にする
    df['Time_in_seconds'] = df['Time_in_seconds'].round().astype(int)

    # スタンプや特殊文字を除去
    def clean_message(x):
        x = re.sub(r':_[A-Z]+:', '', str(x))
        x = re.sub(r'[^\w\s]', '', x)
        return x.strip()

    df['Message'] = df['Message'].apply(clean_message)
    return df


df = preprocess(df)

# GPUが利用可能かチェック
if torch.cuda.is_available():
    device = 0  # GPUを使用
else:
    device = -1  # CPUを使用

# 感情分析パイプラインの作成
sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model="koheiduck/bert-japanese-finetuned-sentiment",
    tokenizer="koheiduck/bert-japanese-finetuned-sentiment",
    device=device
)

# メッセージをバッチで処理し、結果をデータフレームに追加
batch_size = 64  # 適切なバッチサイズを設定
labels = []
scores = []

for i in tqdm.tqdm(range(0, len(df), batch_size)):
    batch_messages = df['Message'].iloc[i:i + batch_size].tolist()
    if len(batch_messages) == 0:
        continue  # 空のバッチはスキップ
    results = sentiment_pipeline(batch_messages)
    print(results)
    batch_labels = [result['label'] for result in results]
    batch_scores = [round(float(result['score']), 5) for result in results]
    labels.extend(batch_labels)
    scores.extend(batch_scores)

df['Label'] = labels
df['Score'] = scores

# ポジティブなコメントのみを抽出
df['Positive'] = df['Label'] == 'ポジティブ'
positive_df = df[df['Positive']].copy()

# 10秒ごとに切り捨て
positive_df['Time_in_10s'] = (positive_df['Time_in_seconds'] // 10) * 10

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
