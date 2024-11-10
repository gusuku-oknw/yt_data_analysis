from transformers import pipeline

model_name = "lxyuan/distilbert-base-multilingual-cased-sentiments-student"
nlp = pipeline('sentiment-analysis', model=model_name, return_all_scores=True)

sentences = [
    "今日はとても良い天気ですね。",
    "昨日の映画は最悪だった。"
]

results = nlp(sentences)
for sentence, result in zip(sentences, results):
    print(f"文章: {sentence}")
    for res in result:
        print(f"ラベル: {res['label']}, スコア: {res['score']:.4f}")
    print(result)
