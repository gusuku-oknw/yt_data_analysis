import pandas as pd
import matplotlib.pyplot as plt
from transformers import pipeline
import torch
import re

# CSVファイルの読み込み
df = pd.read_csv('chat_messages.csv')

# -1 を 0 に置き換え
df = df[df['Time_in_seconds'] >= 0]

# 小数点以下を四捨五入して整数秒にする
df['Time_in_seconds'] = df['Time_in_seconds'].round().astype(int)

# スタンプや特殊文字を除去
df['Message'] = df['Message'].apply(lambda x: re.sub(r':_[A-Z]+:', '', str(x)))
df['Message'] = df['Message'].apply(lambda x: re.sub(r'[^\w\s]', '', x))  # 特殊文字の除去

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


# token_type_ids を除外する設定
def analyze_sentiment(text):
    return sentiment_analyzer(text, truncation=True, padding=True)


print(analyze_sentiment("I love this product!"))  # 例: ポジティブ / ネガティブ


# ポジティブなコメントのみ抽出
def is_positive(text):
    result = sentiment_analyzer(text[:512], truncation=True, padding=True)  # テキストが長い場合は512文字までに制限
    return result[0]['label'] == 'POSITIVE'


# ポジティブなコメントのみを抽出して新しい列に格納
df['Positive'] = df['Message'].apply(lambda x: is_positive(str(x)))

# ポジティブなコメントのみを抽出
positive_df = df[df['Positive']]

# 10秒ごとに切り捨て（例: 21秒 → 20秒、34秒 → 30秒）
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
plt.grid()
plt.tight_layout()
plt.show()
