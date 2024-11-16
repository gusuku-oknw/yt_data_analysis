import pandas as pd
import matplotlib.pyplot as plt

# CSVファイルの読み込み
input_file = 'data/emotion/chat_messages_with_sentiment_bert-finetuned=lFL06DmvdFU.csv.csv'
df = pd.read_csv(input_file)

# 必要な列の存在を確認して追加
if 'Time_in_seconds' in df.columns:
    df['Time_in_10s'] = (df['Time_in_seconds'] // 10) * 10
else:
    raise ValueError("Time_in_seconds 列が存在しません。")

# 感情スコア列からラベルを作成（必要であれば）
emotion_names = ['喜び', '悲しみ', '期待', '驚き', '怒り', '恐れ', '嫌悪', '信頼']
emotion_names = ['positive', 'neutral', 'negative']
# emotion_names_jp = ['ポジティブ', 'ニュートラル', 'ネガティブ']  # 日本語版

if all(col in df.columns for col in emotion_names):
    df['Label'] = df[emotion_names].idxmax(axis=1)
else:
    raise ValueError("感情スコア列が存在しません。")

# 時間ごとの感情データを集計
emotion_counts = df.groupby(['Time_in_10s', 'Label']).size().unstack(fill_value=0)

# グラフをプロット
plt.figure(figsize=(12, 8))
for emotion in emotion_names:
    if emotion in emotion_counts.columns:
        plt.plot(emotion_counts.index, emotion_counts[emotion], marker='o', linestyle='-', label=emotion)

# グラフの設定
plt.xlabel('Time (seconds)', fontsize=12)
plt.ylabel('Comment Count', fontsize=12)
plt.title("Comment Counts per 10 Seconds by Emotion", fontsize=15)
plt.legend(title='Emotion', loc='upper right')
plt.grid(True)
plt.tight_layout()

# プロットを表示
plt.show()
