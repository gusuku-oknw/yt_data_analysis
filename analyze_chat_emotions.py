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
def plot_emotions(df, analysis_methods=["sentiment", "weime", "mlask"], ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))

    for method in analysis_methods:
        if method == "sentiment":
            label_col = "Sentiment_Label"
            counts = df.groupby(['Time_in_seconds', label_col]).size().unstack(fill_value=0)
            for sentiment in sentiment_names:
                if sentiment in counts.columns:
                    ax.plot(counts.index, counts[sentiment], marker='o', linestyle='-', label=f"Sentiment: {sentiment}")
        elif method == "weime":
            label_col = "Weime_Label"
            counts = df.groupby(['Time_in_seconds', label_col]).size().unstack(fill_value=0)
            for emotion in emotion_names:
                if emotion in counts.columns:
                    ax.plot(counts.index, counts[emotion], marker='o', linestyle='-', label=f"Weime: {emotion}")
        elif method == "mlask":
            label_col = "MLAsk_Emotion"
            counts = df.groupby(['Time_in_seconds', label_col]).size().unstack(fill_value=0)
            for emotion in counts.columns:
                ax.plot(counts.index, counts[emotion], marker='o', linestyle='-', label=f"MLAsk: {emotion}")

    ax.set_title('感情別コメント数', fontsize=15)
    ax.legend(title='感情', loc='upper right')
    ax.grid(True)
    plt.xlabel('Time (seconds)')
    plt.ylabel('Number of Comments')
    plt.tight_layout()
    plt.show()


# メイン処理
def main_emotion_analysis(file_path, analysis_methods=["sentiment"], plot_results=True, save_dir="data/emotion"):
    os.makedirs(save_dir, exist_ok=True)

    # CSVファイル読み込みと前処理
    df = pd.read_csv(file_path)
    df = preprocess(df)
    messages = df['Message'].tolist()

    # 分析結果を保存する
    for method in analysis_methods:
        print(f"選択された感情分析方法: {method}")

        # 保存ファイル名を作成
        file_base_name = os.path.splitext(os.path.basename(file_path))[0]
        save_file_name = f"{file_base_name}_analysis.csv"
        save_path = os.path.join(save_dir, save_file_name)

        # ファイルが存在する場合は既存のデータを読み込む
        if os.path.exists(save_path):
            print(f"既存の分析結果を読み込みます: {save_path}")
            df_existing = pd.read_csv(save_path)
        else:
            df_existing = df.copy()

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
                # 修正点: 正しいカラム名を使用
                sentiment_columns = [f"Sentiment_{sentiment.capitalize()}" for sentiment in sentiment_names]
                df_existing['Sentiment_Label'] = df_existing[sentiment_columns].idxmax(axis=1).str.replace('Sentiment_',
                                                                                                           '').str.lower()
            elif method == "weime":
                for i, emotion in enumerate(emotion_names):
                    df_existing[f"Weime_{emotion}"] = [scores[i] for scores in all_scores]
                weime_columns = [f"Weime_{emotion}" for emotion in emotion_names]
                df_existing['Weime_Label'] = df_existing[weime_columns].idxmax(axis=1).str.replace('Weime_', '')

        # 結果をCSVに保存
        df_existing.to_csv(save_path, index=False, encoding='utf-8-sig')
        print(f"{method} の分析結果を保存しました: {save_path}")

    # プロット作成
    if plot_results:
        analysis_results = pd.read_csv(
            os.path.join(save_dir, f"{os.path.splitext(os.path.basename(file_path))[0]}_analysis.csv"))
        plot_emotions(analysis_results, analysis_methods=analysis_methods)

    return


if __name__ == "__main__":
    file_path = 'data/chat_messages/にじさんじ　切り抜き_2024-11-16_18-25-51_videos_processed/4agZGzQLfF8.csv'
    # 複数の感情分析を実行
    main_emotion_analysis(file_path, analysis_methods=["sentiment", "weime", "mlask"], plot_results=True)
