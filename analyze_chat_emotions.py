import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F
import matplotlib
import re
from tqdm import tqdm

# 日本語フォントの設定
matplotlib.rcParams['font.family'] = 'Meiryo'

# ファイルパスの設定
file_path = 'data/chat_messages=SH6HQFhgQ54.csv'  # 解析したいCSVファイルのパスを指定してください

# CSVファイルの読み込み
df = pd.read_csv(file_path)

# データ前処理
def preprocess(df):
    # 時間が-1以上のデータのみを使用
    df = df[df['Time_in_seconds'] >= 0].copy()

    # 時間を整数秒に丸める
    df['Time_in_seconds'] = df['Time_in_seconds'].round().astype(int)

    # メッセージのクリーンアップ
    def clean_message(x):
        x = re.sub(r':_[A-Z]+:', '', str(x))  # 特殊なスタンプの除去
        x = re.sub(r'[^\w\s]', '', x)  # 特殊文字の除去
        return x.strip()

    df['Message'] = df['Message'].apply(clean_message)
    return df

df = preprocess(df)

# デバイスの設定（GPUが利用可能ならGPUを使用）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用デバイス: {device}")

# モデルとトークナイザーの初期化
# 1つ目の感情分析モデル（ポジティブ、ニュートラル、ネガティブ）
checkpoint_sentiment = 'christian-phu/bert-finetuned-japanese-sentiment'
tokenizer_sentiment = AutoTokenizer.from_pretrained(checkpoint_sentiment)
model_sentiment = AutoModelForSequenceClassification.from_pretrained(checkpoint_sentiment)
model_sentiment.to(device)

# 2つ目の感情分析モデル（8つの感情分類）
checkpoint_emotion = './saved_model'  # あなたのモデルのパスに置き換えてください
tokenizer_emotion = AutoTokenizer.from_pretrained(checkpoint_emotion)
model_emotion = AutoModelForSequenceClassification.from_pretrained(checkpoint_emotion)
model_emotion.to(device)

# 感情名の設定
sentiment_names = ['positive', 'neutral', 'negative']
sentiment_names_jp = ['ポジティブ', 'ニュートラル', 'ネガティブ']

emotion_names_jp = ['喜び', '悲しみ', '期待', '驚き', '怒り', '恐れ', '嫌悪', '信頼']

# ソフトマックス関数の定義
def np_softmax(x):
    f_x = np.exp(x) / np.sum(np.exp(x))
    return f_x

# 1つ目の感情分析関数
def sentiment_pipeline(messages):
    results = []
    model_sentiment.eval()

    for message in messages:
        tokens = tokenizer_sentiment(message, truncation=True, max_length=124, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            logits = model_sentiment(**tokens).logits

        probs = F.softmax(logits, dim=-1)[0].cpu().numpy()
        results.append(probs)
    return results

# 2つ目の感情分析関数
def analyze_emotion_batch(messages):
    model_emotion.eval()
    results = []
    for text in messages:
        tokens = tokenizer_emotion(text, truncation=True, return_tensors="pt").to(device)
        with torch.no_grad():
            preds = model_emotion(**tokens)
        prob = np_softmax(preds.logits.cpu().detach().numpy()[0])
        results.append(prob)
    return results

# メッセージをバッチ処理し、結果をデータフレームに追加
batch_size = 64  # 適切なバッチサイズを設定

# 結果を格納するリスト
all_sentiment_scores = []
all_emotion_scores = []

messages = df['Message'].fillna("").tolist()  # メッセージが文字列であることを確認

for i in tqdm(range(0, len(messages), batch_size), desc="バッチ処理中"):
    batch_messages = messages[i:i + batch_size]
    if len(batch_messages) == 0:
        continue

    # 1つ目の感情分析
    batch_sentiment_scores = sentiment_pipeline(batch_messages)
    all_sentiment_scores.extend(batch_sentiment_scores)

    # 2つ目の感情分析
    batch_emotion_scores = analyze_emotion_batch(batch_messages)
    all_emotion_scores.extend(batch_emotion_scores)

# 1つ目の感情スコアをデータフレームに追加
for i, sentiment in enumerate(sentiment_names):
    df[sentiment] = [scores[i] for scores in all_sentiment_scores]

# 2つ目の感情スコアをデータフレームに追加
emotion_df = pd.DataFrame(all_emotion_scores, columns=emotion_names_jp)
df = pd.concat([df.reset_index(drop=True), emotion_df.reset_index(drop=True)], axis=1)

# スコアが最も高い感情をラベルとして追加
df['Sentiment_Label'] = df[sentiment_names].idxmax(axis=1)
df['Emotion_Label'] = df[emotion_names_jp].idxmax(axis=1)

# 時間を10秒単位に切り捨て
df['Time_in_10s'] = (df['Time_in_seconds'] // 10) * 10

# 感情ごとの10秒単位のコメント数を集計
# 1つ目の感情分析結果
sentiment_counts = df.groupby(['Time_in_10s', 'Sentiment_Label']).size().unstack(fill_value=0)

# 2つ目の感情分析結果
emotion_counts = df.groupby(['Time_in_10s', 'Emotion_Label']).size().unstack(fill_value=0)

# 1つ目の感情分析結果をプロット
plt.figure(figsize=(12, 8))
for sentiment in sentiment_names:
    if sentiment in sentiment_counts.columns:
        plt.plot(sentiment_counts.index, sentiment_counts[sentiment], marker='o', linestyle='-', label=sentiment)

plt.xlabel('時間（秒）', fontsize=12)
plt.ylabel('コメント数', fontsize=12)
plt.title("10秒ごとの感情別コメント数（ポジティブ/ニュートラル/ネガティブ）", fontsize=15)
plt.legend(title='感情', loc='upper right')
plt.grid(True)
plt.tight_layout()
plt.show()

# 2つ目の感情分析結果をプロット（必要に応じて感情を選択）
plt.figure(figsize=(12, 8))
for emotion in emotion_names_jp:
    if emotion in emotion_counts.columns:
        plt.plot(emotion_counts.index, emotion_counts[emotion], marker='o', linestyle='-', label=emotion)

plt.xlabel('時間（秒）', fontsize=12)
plt.ylabel('コメント数', fontsize=12)
plt.title("10秒ごとの感情別コメント数（8感情）", fontsize=15)
plt.legend(title='感情', loc='upper right')
plt.grid(True)
plt.tight_layout()
plt.show()

# データフレームをCSVファイルとして保存
file_name = file_path.rsplit('=', 1)[1]
df.to_csv(f'chat_messages_with_sentiment_and_emotion={file_name}.csv', index=False, encoding='utf-8-sig')
print(f"分析結果を 'chat_messages_with_sentiment_and_emotion={file_name}.csv' に保存しました。")
