import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import CheckButtons
import re
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib
from mlask import MLAsk

# 日本語フォントの設定
matplotlib.rcParams['font.family'] = 'Meiryo'


# データの前処理関数
def preprocess(df):
    """
    データフレームを前処理する。

    Parameters:
        df (pd.DataFrame): 読み込んだデータフレーム。

    Returns:
        pd.DataFrame: 前処理済みデータフレーム。
    """
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


# 感情分析の準備
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用デバイス: {device}")

# モデルの準備
checkpoint_sentiment = 'christian-phu/bert-finetuned-japanese-sentiment'
tokenizer_sentiment = AutoTokenizer.from_pretrained(checkpoint_sentiment)
model_sentiment = AutoModelForSequenceClassification.from_pretrained(checkpoint_sentiment).to(device)

checkpoint_weime = './saved_model'
tokenizer_weime = AutoTokenizer.from_pretrained(checkpoint_weime)
model_weime = AutoModelForSequenceClassification.from_pretrained(checkpoint_weime).to(device)

# 感情名
sentiment_names = ['positive', 'neutral', 'negative']
emotion_names = ['Joy', 'Sadness', 'Anticipation', 'Surprise', 'Anger', 'Fear', 'Disgust', 'Trust']


# 感情分析関数
def sentiment_pipeline(messages):
    model_sentiment.eval()
    results = []

    for message in messages:
        tokens = tokenizer_sentiment(message, truncation=True, max_length=124, return_tensors="pt", padding=True).to(
            device)
        with torch.no_grad():
            logits = model_sentiment(**tokens).logits

        probs = F.softmax(logits, dim=-1)[0].cpu().numpy()
        results.append(probs)
    return results


def weime_pipeline(messages):
    model_weime.eval()
    results = []

    for message in messages:
        tokens = tokenizer_weime(message, truncation=True, return_tensors="pt").to(device)
        with torch.no_grad():
            logits = model_weime(**tokens).logits
        probs = F.softmax(logits, dim=-1).cpu().numpy()
        results.append(probs[0])
    return results


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


# 感情分析の実行
def analyze_sentiment(messages, method="sentiment"):
    if method == "sentiment":
        return sentiment_pipeline(messages)
    elif method == "weime":
        return weime_pipeline(messages)
    elif method == "mlask":
        return mlask_pipeline(messages)
    else:
        raise ValueError(f"未知の感情分析方法: {method}")


# プロット作成
def plot_emotions(df, analysis_methods=["sentiment", "weime", "mlask"], save_path=None):
    """
    分析結果をプロットし、左側にチェックボックスを表示して任意の感情を表示/非表示可能にする。
    画像データを保存可能。保存時はグラフの表示をスキップし、チェックボックスを表示しない。

    Parameters:
        df (pd.DataFrame): 分析結果データフレーム。
        analysis_methods (list): 使用する分析方法のリスト。
        save_path (str): グラフ画像を保存するパス（デフォルトはNone）。
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    plot_dict = {}

    for method in analysis_methods:
        if method == "sentiment":
            label_col = "Sentiment_Label"
            if label_col not in df.columns:
                continue
            counts = df.groupby(['Time_in_seconds', label_col]).size().unstack(fill_value=0)
            for sentiment in sentiment_names:
                if sentiment in counts.columns:
                    line, = ax.plot(counts.index, counts[sentiment], marker='o', linestyle='-',
                                    label=f"Sentiment: {sentiment}")
                    plot_dict[f"Sentiment: {sentiment}"] = line
        elif method == "weime":
            label_col = "Weime_Label"
            if label_col not in df.columns:
                continue
            counts = df.groupby(['Time_in_seconds', label_col]).size().unstack(fill_value=0)
            for emotion in emotion_names:
                if emotion in counts.columns:
                    line, = ax.plot(counts.index, counts[emotion], marker='o', linestyle='-', label=f"Weime: {emotion}")
                    plot_dict[f"Weime: {emotion}"] = line
        elif method == "mlask":
            label_col = "MLAsk_Emotion"
            if label_col not in df.columns:
                continue
            # MLAsk_Emotionがカンマ区切りの場合、個別にカウントする
            mlask_expanded = df[['Time_in_seconds', label_col]].copy()
            mlask_expanded[label_col] = mlask_expanded[label_col].str.split(',')
            mlask_expanded = mlask_expanded.explode(label_col)
            mlask_expanded[label_col] = mlask_expanded[label_col].str.strip()
            mlask_counts = mlask_expanded.groupby(['Time_in_seconds', label_col]).size().unstack(fill_value=0)
            for emotion in mlask_counts.columns:
                if emotion.lower() == 'none':
                    continue  # 'none' をプロットしない場合
                line, = ax.plot(mlask_counts.index, mlask_counts[emotion], marker='o', linestyle='-',
                                label=f"MLAsk: {emotion}")
                plot_dict[f"MLAsk: {emotion}"] = line

    ax.set_title('感情別コメント数', fontsize=15)
    ax.legend(title='感情', loc='upper right')
    ax.grid(True)
    plt.xlabel('Time (seconds)')
    plt.ylabel('Number of Comments')
    plt.tight_layout()

    if save_path:
        # 画像を保存して表示しない
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, format="png", bbox_inches="tight")
        print(f"プロット画像を保存しました: {save_path}")
        plt.close(fig)
    else:
        # チェックボックスの表示とグラフのインタラクティブ操作
        plt.subplots_adjust(left=0.35)  # 左側に余裕を持たせる
        rax = plt.axes([0.02, 0.2, 0.3, 0.6])  # [left, bottom, width, height] を調整
        labels = list(plot_dict.keys())
        visibility = [line.get_visible() for line in plot_dict.values()]
        check = CheckButtons(rax, labels, visibility)

        # チェックボックスのコールバック関数
        def func(label):
            line = plot_dict[label]
            line.set_visible(not line.get_visible())
            plt.draw()

        check.on_clicked(func)
        plt.show()


# メイン処理
def main_emotion_analysis(file_path, analysis_methods=["sentiment"], plot_results=True, plot_save=None, save_dir="data/emotion"):
    """
    メイン処理関数:
    1. データ読み込み・前処理
    2. 各分析手法での感情分析（既存結果があればスキップ）
    3. 分析結果保存
    4. プロット表示または保存（オプション）

    Parameters:
        file_path (str): 入力CSVファイルのパス。
        analysis_methods (list): 使用する感情分析手法のリスト（デフォルトは ["sentiment"]）。
        plot_results (bool): プロットを表示するか（デフォルトは True）。
        plot_save (str or None): プロット画像の保存先ファイル名（デフォルトは None）。
        save_dir (str): 分析結果を保存するディレクトリ（デフォルトは "data/emotion"）。

    Returns:
        pd.DataFrame: 最終的な分析結果のデータフレーム。
    """
    os.makedirs(save_dir, exist_ok=True)

    # データ読み込みと前処理
    df = pd.read_csv(str(Path(file_path)).replace("\u3000", "　"))
    df = preprocess(df)
    messages = df['Message'].tolist()

    # 結果を保存するファイルパス生成
    file_base_name = os.path.splitext(os.path.basename(file_path))[0]
    save_file_name = f"{file_base_name}_analysis.csv"
    save_path = os.path.join(save_dir, save_file_name)

    # 既存の結果があれば読み込み、なければオリジナルdfを使用
    if os.path.exists(save_path):
        df_existing = pd.read_csv(save_path)
        print(f"既存の分析結果を読み込みます: {save_path}")
    else:
        df_existing = df.copy()

    # 各分析方法に対して分析実行（既に結果カラムがある場合はスキップ）
    for method in analysis_methods:
        print(f"選択された感情分析方法: {method}")

        # 既に該当する結果が存在するかチェック
        if method == "sentiment" and 'Sentiment_Label' in df_existing.columns:
            print("Sentiment結果は既に存在します。スキップします。")
            continue
        if method == "weime" and 'Weime_Label' in df_existing.columns:
            print("Weime結果は既に存在します。スキップします。")
            continue
        if method == "mlask" and 'MLAsk_Emotion' in df_existing.columns:
            print("MLAsk結果は既に存在します。スキップします。")
            continue

        # 分析実行
        if method == "mlask":
            df_existing['MLAsk_Emotion'] = analyze_sentiment(messages, method=method)
        else:
            all_scores = []
            batch_size = 64
            for i in tqdm(range(0, len(messages), batch_size), desc=f"{method} 感情分析中"):
                batch_messages = messages[i:i + batch_size]
                all_scores.extend(analyze_sentiment(batch_messages, method=method))

            if method == "sentiment":
                for i, sentiment in enumerate(sentiment_names):
                    df_existing[f"Sentiment_{sentiment.capitalize()}"] = [scores[i] for scores in all_scores]
                sentiment_columns = [f"Sentiment_{sentiment.capitalize()}" for sentiment in sentiment_names]
                df_existing['Sentiment_Label'] = df_existing[sentiment_columns].idxmax(axis=1).str.replace('Sentiment_', '').str.lower()

            elif method == "weime":
                for i, emotion in enumerate(emotion_names):
                    df_existing[f"Weime_{emotion}"] = [scores[i] for scores in all_scores]
                weime_columns = [f"Weime_{emotion}" for emotion in emotion_names]
                df_existing['Weime_Label'] = df_existing[weime_columns].idxmax(axis=1).str.replace('Weime_', '')

        # 途中結果を保存
        df_existing.to_csv(save_path, index=False, encoding='utf-8-sig')
        print(f"{method} の分析結果を保存しました: {save_path}")

    # プロット作成または保存
    if plot_results or plot_save:
        analysis_results = pd.read_csv(save_path)
        plot_file = plot_save if plot_save else None
        plot_emotions(analysis_results, analysis_methods=analysis_methods, save_path=plot_file)

    # 最終的なデータフレームをreturn
    return df_existing

if __name__ == "__main__":
    file_path = 'data/chat_messages/にじさんじ　切り抜き_2024-11-16_18-25-51_videos_processed/4agZGzQLfF8.csv'
    print(os.path.basename(file_path))
    # 複数の感情分析を実行
    main_emotion_analysis(
        file_path=file_path,
        analysis_methods=["sentiment", "weime", "mlask"],
        plot_results=False,  # プロットを表示しない
        plot_save=f"data/images/emotion_plot_{os.path.basename(file_path)}.png",  # プロット画像を保存
        save_dir="data/emotion"
    )
