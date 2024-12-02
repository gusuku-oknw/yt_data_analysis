def visualize_combined_with_file(file_path, matches_csv, unmatched_csv):
    """
    セグメント比較と感情分析を1つのプロットに統合して表示する。

    Parameters:
        file_path (str): 感情分析結果を含むCSVファイルパス。
        matches_csv (str): 一致したセグメントのCSVファイルパス。
        unmatched_csv (str): 一致しなかったセグメントのCSVファイルパス。
    """
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter
    import pandas as pd
    import os
    from matplotlib import rcParams

    # 日本語フォントの設定
    rcParams['font.family'] = 'Meiryo'

    # 秒を「時:分:秒」に変換する関数
    def seconds_to_hms(seconds):
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours:02}:{minutes:02}:{secs:02}"

    # CSVデータの読み込み
    df = pd.read_csv(file_path)

    # データのグループ化（`Time_in_10s` 列を使用）
    # 'Time_in_10s' カラムを作成
    df['Time_in_10s'] = (df['Time_in_seconds'] // 10) * 10
    sentiment_counts = df.groupby('Time_in_10s')[['positive', 'neutral', 'negative']].sum()
    emotion_counts = df.groupby('Time_in_10s')[['Joy', 'Sadness', 'Anticipation', 'Surprise', 'Anger', 'Fear', 'Disgust', 'Trust']].sum()

    matches_df = pd.read_csv(matches_csv)

    if os.path.exists(unmatched_csv):
        unmatched_df = pd.read_csv(unmatched_csv)
    else:
        unmatched_df = pd.DataFrame()

    # プロットの設定
    fig, axes = plt.subplots(5, 1, figsize=(12, 20), sharex=True)

    # === セグメント比較（切り抜き音声） ===
    ax1 = axes[0]
    for _, row in matches_df.iterrows():
        ax1.barh(y=0, width=row['clip_end'] - row['clip_start'],
                 left=row['clip_start'], height=0.4, align='center', color='green', label='一致')

    if not unmatched_df.empty:
        for _, row in unmatched_df.iterrows():
            ax1.barh(y=0, width=row['clip_end'] - row['clip_start'],
                     left=row['clip_start'], height=0.4, align='center', color='red', label='不一致')

    ax1.set_title('切り抜き音声のセグメント', fontsize=14)
    ax1.set_xlim(0, matches_df['clip_end'].max())
    ax1.set_yticks([])
    ax1.xaxis.set_major_formatter(FuncFormatter(lambda x, _: seconds_to_hms(x)))
    # ax1.legend(loc='upper right', fontsize=10)

    # === セグメント比較（元音声） ===
    ax2 = axes[1]
    for _, row in matches_df.iterrows():
        ax2.barh(y=0, width=row['source_end'] - row['source_start'],
                 left=row['source_start'], height=0.4, align='center', color='green')

    ax2.set_title('元音声のセグメント', fontsize=14)
    ax2.set_xlim(0, matches_df['source_end'].max())
    ax2.set_yticks([])
    ax2.xaxis.set_major_formatter(FuncFormatter(lambda x, _: seconds_to_hms(x)))

    # === 感情分析（ポジティブ/ニュートラル/ネガティブ） ===
    ax3 = axes[2]
    for column in ['positive', 'neutral', 'negative']:
        ax3.plot(sentiment_counts.index, sentiment_counts[column], marker='o', linestyle='-', label=column)

    ax3.set_title('10秒ごとの感情別コメント数（ポジティブ/ニュートラル/ネガティブ）', fontsize=14)
    ax3.set_ylabel('コメント数', fontsize=12)
    ax3.legend(title='感情', loc='upper right')
    ax3.grid(True)

    # === 感情分析（8分類） ===
    ax4 = axes[3]
    for column in ['Joy', 'Sadness', 'Anticipation', 'Surprise', 'Anger', 'Fear', 'Disgust', 'Trust']:
        ax4.plot(emotion_counts.index, emotion_counts[column], marker='o', linestyle='-', label=column)

    ax4.set_title('10秒ごとの感情別コメント数（8感情）', fontsize=14)
    ax4.set_ylabel('コメント数', fontsize=12)
    ax4.legend(title='感情', loc='upper right')
    ax4.grid(True)

    # === MLAsk感情分析（再構築） ===
    mlask_counts = df.groupby('Time_in_10s')['Sentiment_Label'].value_counts().unstack(fill_value=0)
    ax5 = axes[4]
    for column in mlask_counts.columns:
        ax5.plot(mlask_counts.index, mlask_counts[column], marker='o', linestyle='-', label=column)

    ax5.set_title('10秒ごとの感情別コメント数（MLAsk）', fontsize=14)
    ax5.set_xlabel('時間（時:分:秒）', fontsize=12)
    ax5.set_ylabel('コメント数', fontsize=12)
    ax5.legend(title='感情', loc='upper right')
    ax5.grid(True)

    # レイアウト調整と表示
    plt.tight_layout()
    plt.show()


# 実行例
if __name__ == '__main__':
    file_path = './chat_messages_with_sentiment_and_emotion=YGGLxywB3Tw.csv'
    matches_csv = 'data/compare_CSV/O5Aa-5KqFPqQD8Xd_7-1fNxXj_xM.csv'
    unmatched_csv = 'data/compare_CSV/O5Aa-5KqFPqQD8Xd_7-1fNxXj_xM_unmatched.csv'

    visualize_combined_with_file(file_path, matches_csv, unmatched_csv)
