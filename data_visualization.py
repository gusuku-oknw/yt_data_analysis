import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F
import torch
import re
from tqdm import tqdm

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
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# 感情分析モデルの初期設定
checkpoint = "./saved_model"  # モデル保存先
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
model.to(device)

# 感情ラベル
emotion_names = ['Joy', 'Sadness', 'Anticipation', 'Surprise', 'Anger', 'Fear', 'Disgust', 'Trust']
emotion_names_jp = ['喜び', '悲しみ', '期待', '驚き', '怒り', '恐れ', '嫌悪', '信頼']  # 日本語版


# 感情分析パイプライン関数
def sentiment_pipeline(messages):
    """
    メッセージを入力として感情ラベルとスコアを返す関数
    """
    results = []
    model.eval()

    for message in messages:
        # トークナイズと推論
        tokens = tokenizer(message, truncation=True, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            logits = model(**tokens).logits

        # GPU上でソフトマックス計算
        probs = F.softmax(logits, dim=-1)[0].cpu().numpy()  # 必要に応じてNumPy配列に変換
        results.append(probs)
    return results


# メッセージをバッチで処理し、結果をデータフレームに追加
batch_size = 64  # 適切なバッチサイズを設定
all_scores = []

for i in tqdm(range(0, len(df), batch_size)):
    batch_messages = df['Message'].iloc[i:i + batch_size].tolist()
    if len(batch_messages) == 0:
        continue  # 空のバッチはスキップ
    batch_scores = sentiment_pipeline(batch_messages)
    all_scores.extend(batch_scores)

# 各感情スコアを個別の列として追加
for i, emotion in enumerate(emotion_names):
    df[emotion] = [scores[i] for scores in all_scores]

# 最も高いスコアの感情をラベルとして追加
df['Label'] = df[emotion_names].idxmax(axis=1)

# 時間を10秒単位に切り捨て
df['Time_in_10s'] = (df['Time_in_seconds'] // 10) * 10

# 感情ごとの10秒単位のコメント数を集計
emotion_counts = df.groupby(['Time_in_10s', 'Label']).size().unstack(fill_value=0)

# グラフをプロット
plt.figure(figsize=(12, 8))
for emotion in emotion_names:
    if emotion in emotion_counts.columns:
        plt.plot(emotion_counts.index, emotion_counts[emotion], marker='o', linestyle='-', label=emotion)

plt.xlabel('Time (seconds)', fontsize=12)
plt.ylabel('Comment Count', fontsize=12)
plt.title("Comment Counts per 10 Seconds by Emotion", fontsize=15)
plt.legend(title='Emotion', loc='upper right')
plt.grid(True)
plt.tight_layout()
plt.show()

# 分析結果を新しいCSVファイルとして保存
df.to_csv('chat_messages_with_sentiment.csv', index=False, encoding='utf-8-sig')
