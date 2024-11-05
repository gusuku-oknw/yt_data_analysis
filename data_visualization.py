import pandas as pd
import matplotlib.pyplot as plt

# CSVファイルの読み込み
df = pd.read_csv('chat_messages.csv')

# -1 を 0 に置き換え
df = df[df['Time_in_seconds'] >= 0]

# 小数点以下を四捨五入して整数秒にする
df['Time_in_seconds'] = df['Time_in_seconds'].round().astype(int)

# 10秒ごとに切り捨て（例: 21秒 → 20秒、34秒 → 30秒）
df['Time_in_10s'] = (df['Time_in_seconds'] // 10) * 10

# 10秒ごとにコメント数を集計
ten_secondly_counts = df.groupby('Time_in_10s').size()

# 秒ごとにコメント数を集計
secondly_counts = df.groupby('Time_in_seconds').size()

# 時系列グラフの表示
plt.figure(figsize=(12, 6))
plt.plot(ten_secondly_counts.index, ten_secondly_counts.values, marker='o', linestyle='-', color='steelblue')
plt.xlabel('Date')
plt.ylabel('Comment Count')
plt.title('Daily Comment Counts Over Time')
plt.xticks(rotation=45)
plt.grid()
plt.tight_layout()
plt.show()
