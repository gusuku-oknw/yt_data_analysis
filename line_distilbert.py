import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

# デバイスの設定（GPUが利用可能な場合はGPUを使用）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
# トークナイザーとモデルの読み込み
tokenizer = AutoTokenizer.from_pretrained("line-corporation/line-distilbert-base-japanese", trust_remote_code=True)
model = AutoModelForSequenceClassification.from_pretrained("line-corporation/line-distilbert-base-japanese")

# モデルをデバイスに移動
model.to(device)

# 感情分析パイプラインの作成
sentiment_analyzer = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer, device=device)

# 分析対象の文章
sentences = [
    "今日はとても良い天気ですね。",
    "昨日の映画は最悪だった。"
]

# 感情分析の実行
results = sentiment_analyzer(sentences)
print(results)

# 結果の表示
for sentence, result in zip(sentences, results):
    print(f"文章: {sentence}")
    print(f"ラベル: {result['label']}, スコア: {result['score']:.4f}")
    print()
