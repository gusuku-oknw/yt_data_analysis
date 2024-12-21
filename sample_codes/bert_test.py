from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline

# 文脈ベースのモデル
tokenizer = AutoTokenizer.from_pretrained("cl-tohoku/bert-base-japanese")
model = AutoModelForTokenClassification.from_pretrained("cl-tohoku/bert-base-japanese")

# pipelineを初期化
nlp = pipeline("ner", model=model, tokenizer=tokenizer)

text = "これはテストですこれは2つ目の文です！最後の文です？"

# 文脈で分割する関数
def split_by_bert_pipeline(text):
    tokens = tokenizer.tokenize(text)
    chunks = []
    current_chunk = ""
    for token in tokens:
        current_chunk += token
        if token in ["。", "！", "？"]:  # 句点があれば文を切る
            chunks.append(current_chunk)
            current_chunk = ""
    if current_chunk:
        chunks.append(current_chunk)
    return chunks

# 実行
chunks = split_by_bert_pipeline(text)
for chunk in chunks:
    print(chunk)
