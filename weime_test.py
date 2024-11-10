import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import matplotlib

# 必要な初期設定
checkpoint = "./saved_model"  # モデル保存先
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
device = torch.device("cuda")
model.to(device)

# 感情名リスト (日本語)
emotion_names_jp = ['喜び', '悲しみ', '期待', '驚き', '怒り', '恐れ', '嫌悪', '信頼']  # 日本語版


# ソフトマックス関数
def np_softmax(x):
    f_x = np.exp(x) / np.sum(np.exp(x))
    return f_x


# 日本語フォントの設定
# フォントを設定 (Windows用の例: "Meiryo")
matplotlib.rcParams['font.family'] = 'Meiryo'


# 感情分析とグラフ描画関数
def analyze_emotion(text, show_fig=False):
    """
    入力文に対する感情予測を行い、必要に応じて棒グラフを描画する。

    Args:
        text (str): 入力文。
        show_fig (bool): 棒グラフを描画するかどうか。デフォルトはFalse。
    """
    # 推論モードを有効化
    model.eval()

    # 入力データ変換 + 推論
    tokens = tokenizer(text, truncation=True, return_tensors="pt")
    tokens.to(model.device)
    with torch.no_grad():
        preds = model(**tokens)
    prob = np_softmax(preds.logits.cpu().detach().numpy()[0])
    out_dict = {n: p for n, p in zip(emotion_names_jp, prob)}

    # 棒グラフを描画
    if show_fig:
        plt.figure(figsize=(8, 3))
        df = pd.DataFrame(out_dict.items(), columns=['name', 'prob'])
        sns.barplot(x='name', y='prob', data=df)
        plt.title('入力文: ' + text, fontsize=15)
        plt.ylabel('確率', fontsize=12)
        plt.xlabel('感情', fontsize=12)
        plt.xticks(rotation=45)  # 日本語が切れないように回転を追加
        plt.show()
    else:
        print(out_dict)


# 使用例
analyze_emotion('昨日の映画は最悪だった。', show_fig=True)
