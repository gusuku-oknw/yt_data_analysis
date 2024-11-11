import torch
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer

# デバイスを指定（GPUが利用可能ならCUDAを使用）
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 事前学習済みの日本語感情分析モデルとそのトークナイザをロード
model = AutoModelForSequenceClassification.from_pretrained('christian-phu/bert-finetuned-japanese-sentiment')
model.to(device)  # モデルをGPUに移動
tokenizer = AutoTokenizer.from_pretrained('christian-phu/bert-finetuned-japanese-sentiment', model_max_length=512)

# 分析対象となるテキストのリスト
texts = ['給料が高くて満足しています。', '給料低すぎるだろ！', '可もなく不可もなく']

# 各テキストに対して感情分析を実行
for text in texts:
    print('*' * 50)
    # テキストをトークナイズしてGPUに移動
    inputs = tokenizer(text, padding=True, truncation=True, return_tensors='pt', max_length=512).to(device)
    outputs = model(**inputs)
    logits = outputs.logits

    # ロジットを確率に変換
    probabilities = torch.softmax(logits, dim=1)[0]

    # 最も高い確率の感情ラベルを取得
    sentiment_label = model.config.id2label[torch.argmax(probabilities).item()]

    print('テキスト：{}'.format(text))
    print('感情：{}'.format(sentiment_label))

    # positiveまたはnegativeの場合はその確率を表示、neutralの場合はpositiveとnegativeの最大値を表示
    if (sentiment_label == 'positive') or (sentiment_label == 'negative'):
        print('感情スコア：{}'.format(max(probabilities)))
    else:
        print('感情スコア：{}'.format(max(probabilities[0], probabilities[2])))

# 出力結果
# **************************************************
# テキスト：給料が高くて満足しています。
# 感情：positive
# 感情スコア：0.9992052912712097
# **************************************************
# テキスト：給料低すぎるだろ！
# 感情：negative
# 感情スコア：0.9936193823814392
# **************************************************
# テキスト：可もなく不可もなく
# 感情：neutral
# 感情スコア：0.0014821902150288224
