import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib
from mlask import MLAsk

# 日本語フォントの設定
matplotlib.rcParams['font.family'] = 'Meiryo'

# ファイルパスの設定
file_path = 'data/chat_messages/ホロライブ　切り抜き/YGGLxywB3Tw.csv'  # 解析したいCSVファイルのパスを指定してください

# データの読み込みと前処理
def preprocess(df):
    # 時間が-1以上のデータのみを使用
    df = df[df['Time_in_seconds'] >= 0].copy()

    # 時間を整数秒に丸める
    df['Time_in_seconds'] = df['Time_in_seconds'].round().astype(int)

    # メッセージのクリーンアップ
    def clean_message(x):
        x = re.sub(r':_[A-Z]+:', '', str(x))
        x = re.sub(r'[^\w\s]', '', x)
        return x.strip()

    df['Message'] = df['Message'].apply(clean_message)
    return df


df = pd.read_csv(file_path)
df = preprocess(df)

# デバイスの設定
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用デバイス: {device}")

# モデルの準備
checkpoint_sentiment = 'christian-phu/bert-finetuned-japanese-sentiment'
tokenizer_sentiment = AutoTokenizer.from_pretrained(checkpoint_sentiment)
model_sentiment = AutoModelForSequenceClassification.from_pretrained(checkpoint_sentiment).to(device)

checkpoint_emotion = './saved_model'
tokenizer_emotion = AutoTokenizer.from_pretrained(checkpoint_emotion)
model_emotion = AutoModelForSequenceClassification.from_pretrained(checkpoint_emotion).to(device)

# 感情名
sentiment_names = ['positive', 'neutral', 'negative']
sentiment_names_jp = ['ポジティブ', 'ニュートラル', 'ネガティブ']
emotion_names = ['Joy', 'Sadness', 'Anticipation', 'Surprise', 'Anger', 'Fear', 'Disgust', 'Trust']
emotion_names_jp = ['喜び', '悲しみ', '期待', '驚き', '怒り', '恐れ', '嫌悪', '信頼']


# 感情分析関数
def sentiment_pipeline(messages):
    model_sentiment.eval()
    results = []

    for message in messages:
        tokens = tokenizer_sentiment(message, truncation=True, max_length=124, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            logits = model_sentiment(**tokens).logits

        probs = F.softmax(logits, dim=-1)[0].cpu().numpy()
        results.append(probs)
    return results


def emotion_pipeline(messages):
    model_emotion.eval()
    results = []

    for message in messages:
        tokens = tokenizer_emotion(message, truncation=True, return_tensors="pt").to(device)
        with torch.no_grad():
            logits = model_emotion(**tokens).logits
        probs = np.exp(logits.cpu().numpy()) / np.exp(logits.cpu().numpy()).sum(axis=-1, keepdims=True)
        results.append(probs[0])
    return results


# MLAskによる感情分析
def mlask_pipeline(messages):
    analyzer = MLAsk()
    results = []
    for message in messages:
        if not message or message.strip() == "":
            results.append('none')
            continue
        analysis = analyzer.analyze(message)
        if analysis['emotion']:
            emotions = ",".join(analysis['emotion'].keys())
            results.append(emotions)
        else:
            results.append('none')
    return results


# バッチ処理
batch_size = 64
messages = df['Message'].tolist()
all_sentiment_scores = []
all_emotion_scores = []

for i in tqdm(range(0, len(messages), batch_size), desc="感情分析中"):
    batch_messages = messages[i:i + batch_size]
    all_sentiment_scores.extend(sentiment_pipeline(batch_messages))
    all_emotion_scores.extend(emotion_pipeline(batch_messages))

# データフレームに結果を追加
for i, sentiment in enumerate(sentiment_names):
    df[sentiment] = [scores[i] for scores in all_sentiment_scores]

# 2つ目の感情スコアをデータフレームに追加
emotion_df = pd.DataFrame(all_emotion_scores, columns=emotion_names)
df = pd.concat([df.reset_index(drop=True), emotion_df.reset_index(drop=True)], axis=1)

# スコアが最も高い感情をラベルとして追加
df['Sentiment_Label'] = df[sentiment_names].idxmax(axis=1)
df['Emotion_Label'] = df[emotion_names].idxmax(axis=1)

# MLAsk感情分析の結果を追加
print("MLAskで感情分析を実行中...")
df['MLAsk_Emotion'] = mlask_pipeline(messages)

# MLAsk感情のプロット用データ作成
mlask_counts = df.groupby(['Time_in_seconds', 'MLAsk_Emotion']).size().unstack(fill_value=0)

# プロット
fig, axes = plt.subplots(3, 1, figsize=(12, 15), sharex=True)

# === ポジティブ/ニュートラル/ネガティブのプロット ===
sentiment_counts = df.groupby(['Time_in_seconds', 'Sentiment_Label']).size().unstack(fill_value=0)
for sentiment in sentiment_names:
    if sentiment in sentiment_counts.columns:
        axes[0].plot(sentiment_counts.index, sentiment_counts[sentiment], marker='o', linestyle='-', label=sentiment)
axes[0].set_title('感情別コメント数（ポジティブ/ニュートラル/ネガティブ）', fontsize=15)
axes[0].legend(title='感情', loc='upper right')
axes[0].grid(True)

# === 8分類感情のプロット ===
emotion_counts = df.groupby(['Time_in_seconds', 'Emotion_Label']).size().unstack(fill_value=0)
for emotion in emotion_names:
    if emotion in emotion_counts.columns:
        axes[1].plot(emotion_counts.index, emotion_counts[emotion], marker='o', linestyle='-', label=emotion)
axes[1].set_title('感情別コメント数（8分類）', fontsize=15)
axes[1].legend(title='感情', loc='upper right')
axes[1].grid(True)

# === MLAsk感情のプロット ===
if not mlask_counts.empty:  # データが空でない場合のみプロット
    for mlask_emotion in mlask_counts.columns:
        axes[2].plot(mlask_counts.index, mlask_counts[mlask_emotion], marker='o', linestyle='-', label=mlask_emotion)
    axes[2].set_title('MLAsk感情別コメント数', fontsize=15)
    axes[2].legend(title='感情', loc='upper right')
    axes[2].grid(True)
else:
    print("MLAsk感情データがありません。プロットをスキップします。")

# レイアウト調整と表示
plt.tight_layout()
plt.show()

# データフレームをCSVファイルとして保存
file_name = os.path.splitext(os.path.basename(file_path))[0]  # 拡張子を除いたファイル名を取得
df.to_csv(f'./chat_messages_with_sentiment_and_emotion={file_name}.csv', index=False, encoding='utf-8-sig')
print(f"分析結果を 'chat_messages_with_sentiment_and_emotion={file_name}.csv' に保存しました。")
