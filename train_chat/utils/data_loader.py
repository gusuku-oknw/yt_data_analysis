import os
import pandas as pd
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer


class ChatDataset(Dataset):
    def __init__(self, data, labels, tokenizer):
        self.data = data
        self.labels = labels
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data[idx]
        label = self.labels[idx]
        encoded = self.tokenizer(text, padding="max_length", truncation=True, max_length=50, return_tensors="pt")
        return encoded["input_ids"].squeeze(0), label


def load_data_from_dirs(directories):
    """
    指定されたディレクトリリストからCSVファイルをロードし、データを統合する。

    Parameters:
        directories (list): ディレクトリのリスト。

    Returns:
        pd.DataFrame: 統合されたデータフレーム。
    """
    all_data = []
    for directory in directories:
        if not os.path.exists(directory):
            print(f"Directory does not exist: {directory}")
            continue

        for filename in os.listdir(directory):
            if filename.endswith(".csv"):
                file_path = os.path.join(directory, filename)
                try:
                    print(f"Loading data from: {file_path}")
                    df = pd.read_csv(file_path)
                    all_data.append(df)
                except Exception as e:
                    print(f"Error loading {file_path}: {e}")

    if all_data:
        combined_data = pd.concat(all_data, ignore_index=True)
        return combined_data
    else:
        raise ValueError("No valid data found in the specified directories.")


def load_data(train_dirs):
    """
    複数ディレクトリからデータをロードし、学習用と評価用に分割する。

    Parameters:
        train_dirs (list): 学習データが格納されたディレクトリのリスト。

    Returns:
        tuple: (train_data, val_data)
    """
    # ディレクトリからデータをロード
    combined_data = load_data_from_dirs(train_dirs)

    # テキストとラベルを抽出
    if 'text' not in combined_data.columns or 'label' not in combined_data.columns:
        raise KeyError("Combined data must have 'text' and 'label' columns.")

    texts = combined_data['text'].tolist()
    labels = combined_data['label'].tolist()

    # データを学習用と評価用に分割
    train_texts, val_texts, train_labels, val_labels = train_test_split(texts, labels, test_size=0.2, random_state=42)

    # トークナイザーの初期化
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # Datasetの作成
    train_data = ChatDataset(train_texts, train_labels, tokenizer)
    val_data = ChatDataset(val_texts, val_labels, tokenizer)

    return train_data, val_data
