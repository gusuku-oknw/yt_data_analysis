import pandas as pd
import matplotlib.pyplot as plt

# CSVファイルの読み込み
df = pd.read_csv('chat_messages.csv')

# -1 を 0 に置き換え
df = df[df['Time_in_seconds'] >= 0]

# 小数点以下を四捨五入して整数秒にする
df['Time_in_seconds'] = df['Time_in_seconds'].round().astype(int)

# 秒をエポック時間として日時に変換
df['Date'] = pd.to_datetime(df['Time_in_seconds'], unit='s', origin='unix')

# 秒ごとにコメント数を集計
secondly_counts = df.groupby('Time_in_seconds').size()

# 時系列グラフの表示
plt.figure(figsize=(12, 6))
plt.plot(secondly_counts.index, secondly_counts.values, marker='o', linestyle='-', color='steelblue')
plt.xlabel('Date')
plt.ylabel('Comment Count')
plt.title('Daily Comment Counts Over Time')
plt.xticks(rotation=45)
plt.grid()
plt.tight_layout()
plt.show()
