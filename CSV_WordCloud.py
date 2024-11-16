import pandas as pd
from wordcloud import WordCloud
import MeCab
import matplotlib.pyplot as plt

# CSVファイルの読み込み
df = pd.read_csv('data/chat_messages/chat_messages=SH6HQFhgQ54.csv')

# 全メッセージからテキストを結合
all_text = ' '.join(df['Message'].dropna())  # NaNのメッセージを除外して結合

# MeCabの初期化
mecabTagger = MeCab.Tagger()

# 名詞の出現回数をカウント
noun_count = {}

node = mecabTagger.parseToNode(all_text)
while node:
    word = node.surface
    hinshi = node.feature.split(",")[0]
    if hinshi == "名詞":
        if word in noun_count:
            noun_count[word] += 1
        else:
            noun_count[word] = 1
    node = node.next

# 単語リストが空でないことを確認
if not noun_count:
    print("単語リストが空です。データに問題がないか確認してください。")
else:
    # ワードクラウドの生成
    wordcloud = WordCloud(
        font_path='C:/Windows/Fonts/meiryo.ttc',  # 使用する日本語フォントのパス
        background_color='white',
        width=800,
        height=400,
        max_words=200,
        contour_width=1,
        contour_color='steelblue'
    )

    # ワードクラウドを単語の頻度から生成
    wordcloud.generate_from_frequencies(noun_count)

    # ワードクラウドの表示
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.show()
