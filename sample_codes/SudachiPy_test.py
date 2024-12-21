from sudachipy import dictionary
from sudachipy import tokenizer

# 辞書オブジェクトを作成
tokenizer_obj = dictionary.Dictionary().create()
mode = tokenizer.Tokenizer.SplitMode.C

text = "これはテストですこれは2つ目の文です！最後の文です？"

# 文章区切り処理
sentences = []
sentence = ''
for token in tokenizer_obj.tokenize(text, mode):
    sentence += token.surface()  # 修正: token.surface() を呼び出す
    if token.surface() in ['。', '！', '？']:
        sentences.append(sentence)
        sentence = ''  # 次の文のために初期化
if sentence:  # 最後の文をリストに追加
    sentences.append(sentence)

# 結果を表示
for s in sentences:
    print(s)
