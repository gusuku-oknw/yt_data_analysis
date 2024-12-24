import os
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib
from mlask import MLAsk
import re

matplotlib.use('Agg')  # GUIを使用しないバックエンド
matplotlib.rcParams['font.family'] = 'Meiryo'  # 日本語フォント設定


class EmotionAnalyzer:
    def __init__(self, output_dir="../data/emotion"):
        """
        初期化と感情分析モデルのロード。

        Parameters:
            output_dir (str): 結果を保存するディレクトリ。
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # 使用するデバイス
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"使用デバイス: {self.device}")

        # モデルのロード
        self.sentiment_model, self.sentiment_tokenizer = self._load_model('christian-phu/bert-finetuned-japanese-sentiment')
        self.weime_model, self.weime_tokenizer = self._load_model('../saved_model')

        # 感情名
        self.sentiment_labels = ['positive', 'neutral', 'negative']
        self.weime_labels = ['Joy', 'Sadness', 'Anticipation', 'Surprise', 'Anger', 'Fear', 'Disgust', 'Trust']

    def _load_model(self, checkpoint):
        """
        モデルとトークナイザーをロードする。

        Parameters:
            checkpoint (str): モデルのチェックポイント。

        Returns:
            tuple: モデルとトークナイザー。
        """
        tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        model = AutoModelForSequenceClassification.from_pretrained(checkpoint).to(self.device)
        return model, tokenizer

    @staticmethod
    def _preprocess_data(file_path):
        """
        CSVファイルを読み込み、データを前処理する。

        Parameters:
            file_path (str): 入力CSVファイルのパス。

        Returns:
            pd.DataFrame: 前処理済みデータフレーム。
        """
        df = pd.read_csv(file_path)
        df = df[df['Time_in_seconds'] >= 0].copy()
        df['Time_in_seconds'] = df['Time_in_seconds'].round().astype(int)

        def clean_message(message):
            message = re.sub(r':_[A-Z]+:', '', str(message))
            message = re.sub(r'[^\w\s]', '', message)
            return message.strip()

        df['Message'] = df['Message'].apply(clean_message)
        return df

    def _analyze_sentiment(self, messages, method, batch_size=64):
        """
        メッセージリストを指定された感情分析モデルで処理する。

        Parameters:
            messages (list): メッセージリスト。
            method (str): 使用する感情分析 ("sentiment" | "weime" | "mlask")。
            batch_size (int): バッチサイズ。

        Returns:
            list: 分析結果。
        """
        if method == "mlask":
            analyzer = MLAsk()
            return [",".join(analyzer.analyze(msg).get('emotion', {}).keys()) if msg else 'none' for msg in messages]

        model, tokenizer = (self.sentiment_model, self.sentiment_tokenizer) if method == "sentiment" else (self.weime_model, self.weime_tokenizer)
        model.eval()
        results = []

        for i in tqdm(range(0, len(messages), batch_size), desc=f"{method}感情分析中"):
            batch = messages[i:i + batch_size]
            tokens = tokenizer(batch, truncation=True, max_length=124, padding=True, return_tensors="pt").to(self.device)
            with torch.no_grad():
                logits = model(**tokens).logits
            probs = F.softmax(logits, dim=-1).cpu().numpy()
            results.extend(probs)

        return results

    def analyze_file(self, file_path, analysis_methods=None, plot_results=False, plot_save_path=None):
        """
        ファイルを感情分析し、結果を保存またはプロットする。

        Parameters:
            file_path (str): 入力CSVファイルのパス。
            analysis_methods (list): 使用する分析手法 (デフォルト: ['sentiment'])。
            plot_results (bool): プロットを表示するか。
            plot_save_path (str): プロット画像の保存先パス。

        Returns:
            pd.DataFrame: 分析結果。
        """
        if analysis_methods is None:
            analysis_methods = ['sentiment']

        df = self._preprocess_data(file_path)
        save_path = os.path.join(self.output_dir, f"{Path(file_path).stem}_analysis.csv")

        # 既存結果をロード
        if os.path.exists(save_path):
            print(f"既存結果をロード: {save_path}")
            result_df = pd.read_csv(save_path)
        else:
            result_df = df.copy()

        # 各分析を実行
        for method in analysis_methods:
            if f"{method}_Label" in result_df.columns:
                print(f"{method}結果は既に存在します。スキップします。")
                continue

            results = self._analyze_sentiment(df['Message'].tolist(), method)
            if method in ["sentiment", "weime"]:
                labels = self.sentiment_labels if method == "sentiment" else self.weime_labels
                for i, label in enumerate(labels):
                    result_df[f"{method}_{label}"] = [res[i] for res in results]
                result_df[f"{method}_Label"] = result_df.filter(like=f"{method}_").idxmax(axis=1).str.replace(f"{method}_", '')

            elif method == "mlask":
                result_df["mlask_Emotion"] = results

        # 結果を保存
        result_df.to_csv(save_path, index=False, encoding="utf-8-sig")
        print(f"分析結果を保存: {save_path}")

        # プロットを生成
        if plot_results or plot_save_path:
            self._plot_results(result_df, analysis_methods, plot_save_path)

        return result_df

    @staticmethod
    def _plot_results(df, analysis_methods, save_path=None):
        """
        分析結果をプロットする。

        Parameters:
            df (pd.DataFrame): 分析結果データフレーム。
            analysis_methods (list): 使用した分析手法。
            save_path (str): 保存先パス。
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        for method in analysis_methods:
            label_col = f"{method}_Label"
            if label_col not in df.columns:
                continue
            counts = df.groupby(['Time_in_seconds', label_col]).size().unstack(fill_value=0)
            counts.plot(ax=ax, marker='o', linestyle='-')
        ax.set_title("感情分析結果")
        ax.set_xlabel("時間 (秒)")
        ax.set_ylabel("コメント数")
        ax.legend()
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
            print(f"プロット保存: {save_path}")
        else:
            plt.show()


# 使用例
if __name__ == "__main__":
    analyzer = EmotionAnalyzer(output_dir="../data/emotion")
    analyzer.analyze_file(
        file_path="../data/chat_messages/sample.csv",
        analysis_methods=["sentiment", "weime", "mlask"],
        plot_results=True,
        plot_save_path="../data/images/emotion_plot.png"
    )
