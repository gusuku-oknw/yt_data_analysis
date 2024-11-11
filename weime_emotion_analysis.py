import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import matplotlib
from tqdm import tqdm

# 必要な初期設定
checkpoint = "./saved_model"  # 事前学習済みモデルの保存先
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 感情名リスト (日本語)
emotion_names_jp = ['喜び', '悲しみ', '期待', '驚き', '怒り', '恐れ', '嫌悪', '信頼']


# ソフトマックス関数
def np_softmax(x):
    f_x = np.exp(x) / np.sum(np.exp(x))
    return f_x


# 日本語フォントの設定
matplotlib.rcParams['font.family'] = 'Meiryo'


# 感情分析関数
def analyze_emotion_batch(messages):
    """
    テキストのリストを入力として感情スコアを計算する。

    Args:
        messages (list): 感情分析を行うテキストのリスト。

    Returns:
        list: 各テキストの感情スコア（辞書形式）。
    """
    model.eval()
    results = []
    for text in tqdm(messages, desc="感情分析中"):
        # トークナイズと推論
        tokens = tokenizer(text, truncation=True, return_tensors="pt")
        tokens.to(model.device)
        with torch.no_grad():
            preds = model(**tokens)
        prob = np_softmax(preds.logits.cpu().detach().numpy()[0])
        results.append({emotion: p for emotion, p in zip(emotion_names_jp, prob)})
    return results


# メイン処理
if __name__ == "__main__":
    # CSVファイルの読み込み
    input_file = "chat_messages=lFL06DmvdFU.csv"  # 入力CSVファイル名
    output_file = "chat_messages_with_emotion.csv"  # 出力CSVファイル名

    file_name = input_file.rsplit('=', 1)[1]

    df = pd.read_csv(input_file)

    # データ前処理: 必要ならばテキスト列を指定
    text_column = "Message"  # テキストが含まれる列名
    if text_column not in df.columns:
        raise ValueError(f"'{text_column}' 列がCSVファイルに存在しません。")

    # 感情分析の実行
    messages = df[text_column].fillna("").tolist()  # NaNを空文字列に変換
    emotion_results = analyze_emotion_batch(messages)

    # 分析結果をデータフレームに変換
    emotion_df = pd.DataFrame(emotion_results)
    df = pd.concat([df, emotion_df], axis=1)  # 元のデータフレームに感情スコアを結合

    # CSVファイルとして保存
    df.to_csv(f'{output_file}={file_name}', index=False, encoding="utf-8-sig")
    print(f"分析結果を {output_file} として保存しました。")
