import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer
from datasets import Dataset
from evaluate import load

# データ読み込み
df_wrime = pd.read_table('wrime-ver1.tsv')

# モデル設定
checkpoint = 'cl-tohoku/bert-base-japanese-whole-word-masking'
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=8)

emotion_names = ['Joy', 'Sadness', 'Anticipation', 'Surprise', 'Anger', 'Fear', 'Disgust', 'Trust']
emotion_names_jp = ['喜び', '悲しみ', '期待', '驚き', '怒り', '恐れ', '嫌悪', '信頼']  # 日本語版

# 客観感情の平均
df_wrime['readers_emotion_intensities'] = df_wrime.apply(
    lambda x: [x['Avg. Readers_' + name] for name in emotion_names], axis=1)

is_target = df_wrime['readers_emotion_intensities'].map(lambda x: max(x) >= 2)
df_wrime_target = df_wrime[is_target]

df_groups = df_wrime_target.groupby('Train/Dev/Test')
df_train = df_groups.get_group('train')
df_test = pd.concat([df_groups.get_group('dev'), df_groups.get_group('test')])

target_columns = ['Sentence', 'readers_emotion_intensities']
train_dataset = Dataset.from_pandas(df_train[target_columns])
test_dataset = Dataset.from_pandas(df_test[target_columns])


# トークナイズとラベル正規化
def normalize_intensities(intensities):
    total = np.sum(intensities)
    return intensities / total if total > 0 else np.zeros_like(intensities)


def tokenize_function(batch):
    tokenized_batch = tokenizer(batch['Sentence'], truncation=True, padding='max_length')
    tokenized_batch['labels'] = torch.tensor(
        [normalize_intensities(x) for x in batch['readers_emotion_intensities']], dtype=torch.float32
    )
    return tokenized_batch


train_tokenized_dataset = train_dataset.map(tokenize_function, batched=True)
test_tokenized_dataset = test_dataset.map(tokenize_function, batched=True)

# 評価指標
metric = load("accuracy")


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    label_ids = np.argmax(labels, axis=-1)  # すでにnumpy.ndarray形式なので.numpy()は不要
    return metric.compute(predictions=predictions, references=label_ids)


# 訓練設定
training_args = TrainingArguments(
    output_dir="test_trainer",
    per_device_train_batch_size=8,
    num_train_epochs=1.0,
    evaluation_strategy="steps",
    eval_steps=200,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_tokenized_dataset,
    eval_dataset=test_tokenized_dataset,
    compute_metrics=compute_metrics,
)

trainer.train()

# 保存
trainer.save_model("saved_model")  # モデルを保存
tokenizer.save_pretrained("saved_model")
