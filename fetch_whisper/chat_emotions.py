import os
from pathlib import Path
import numpy as np
import pandas as pd
import re
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F
from tqdm import tqdm
from mlask import MLAsk


class EmotionAnalyzer:
    def __init__(
        self,
        checkpoint_sentiment="christian-phu/bert-finetuned-japanese-sentiment",
        checkpoint_weime="../saved_model",
        save_dir="../data/emotion",
    ):
        """
        コンストラクタ: モデルのロードやデバイスの設定などを行う
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"使用デバイス: {self.device}")

        # モデルの準備 (sentiment)
        self.checkpoint_sentiment = checkpoint_sentiment
        self.tokenizer_sentiment = AutoTokenizer.from_pretrained(self.checkpoint_sentiment)
        self.model_sentiment = AutoModelForSequenceClassification.from_pretrained(self.checkpoint_sentiment).to(
            self.device
        )

        # モデルの準備 (weime)
        self.checkpoint_weime = checkpoint_weime
        self.tokenizer_weime = AutoTokenizer.from_pretrained(self.checkpoint_weime)
        self.model_weime = AutoModelForSequenceClassification.from_pretrained(self.checkpoint_weime).to(self.device)

        # 感情名など
        self.sentiment_names = ["positive", "neutral", "negative"]
        self.emotion_names = ["Joy", "Sadness", "Anticipation", "Surprise", "Anger", "Fear", "Disgust", "Trust"]

        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)

    # -----------------------------------
    # データの前処理メソッド
    # -----------------------------------
    def preprocess(self, df):
        """
        データフレームを前処理する。

        Parameters:
            df (pd.DataFrame): 読み込んだデータフレーム。

        Returns:
            pd.DataFrame: 前処理済みデータフレーム。
        """
        # 時間が-1以上のデータのみを使用
        df = df[df["Time_in_seconds"] >= 0].copy()

        # 時間を整数秒に丸める
        df["Time_in_seconds"] = df["Time_in_seconds"].round().astype(int)

        # メッセージのクリーンアップ関数
        def clean_message(x):
            x = re.sub(r":_[A-Z]+:", "", str(x))
            x = re.sub(r"[^\w\s]", "", x)
            return x.strip()

        df["Message"] = df["Message"].apply(clean_message)
        return df

    # -----------------------------------
    # 個別パイプライン
    # -----------------------------------
    def sentiment_pipeline(self, messages):
        """
        BERTの日本語sentimentモデルを用いた感情分析
        """
        self.model_sentiment.eval()
        results = []

        for message in messages:
            tokens = self.tokenizer_sentiment(
                message, truncation=True, max_length=124, return_tensors="pt", padding=True
            ).to(self.device)

            with torch.no_grad():
                logits = self.model_sentiment(**tokens).logits

            probs = F.softmax(logits, dim=-1)[0].cpu().numpy()
            results.append(probs)
        return results

    def weime_pipeline(self, messages):
        """
        Weime (カスタムモデル) を用いた感情分類
        """
        self.model_weime.eval()
        results = []

        for message in messages:
            tokens = self.tokenizer_weime(message, truncation=True, return_tensors="pt").to(self.device)
            with torch.no_grad():
                logits = self.model_weime(**tokens).logits
            probs = F.softmax(logits, dim=-1).cpu().numpy()
            results.append(probs[0])
        return results

    def mlask_pipeline(self, messages):
        """
        MLAsk を用いた日本語文章感情分析
        """
        analyzer = MLAsk()
        results = []
        for message in messages:
            if not message or message.strip() == "":
                results.append("none")
                continue
            analysis = analyzer.analyze(message)
            if analysis["emotion"]:
                emotions = ",".join(analysis["emotion"].keys())
                results.append(emotions)
            else:
                results.append("none")
        return results

    # -----------------------------------
    # 統合された感情分析メソッド
    # -----------------------------------
    def analyze_sentiment(self, messages, method="sentiment"):
        """
        指定したmethod("sentiment" or "weime" or "mlask")で感情分析を実行
        """
        if method == "sentiment":
            return self.sentiment_pipeline(messages)
        elif method == "weime":
            return self.weime_pipeline(messages)
        elif method == "mlask":
            return self.mlask_pipeline(messages)
        else:
            raise ValueError(f"未知の感情分析方法: {method}")

    # -----------------------------------
    # メイン処理
    # -----------------------------------
    def analysis_emotion(self, file_path, analysis_methods=["sentiment"]):
        """
        メイン処理関数:
        1. データ読み込み・前処理
        2. 各分析手法での感情分析（既存結果があればスキップ）
        3. 分析結果保存
        4. 最終的なデータフレームを返却

        Parameters:
            file_path (str): 入力CSVファイルのパス。
            analysis_methods (list): 使用する感情分析手法のリスト（デフォルトは ["sentiment"]）。

        Returns:
            pd.DataFrame: 最終的な分析結果のデータフレーム。
        """
        # データ読み込みと前処理
        df = pd.read_csv(str(Path(file_path)).replace("\u3000", "　"))
        df = self.preprocess(df)
        messages = df["Message"].tolist()

        # 結果を保存するファイルパス生成
        file_base_name = os.path.splitext(os.path.basename(file_path))[0]
        save_file_name = f"{file_base_name}_analysis.csv"
        save_path = os.path.join(self.save_dir, save_file_name)

        # 既存の結果があれば読み込み、なければオリジナルdfを使用
        if os.path.exists(save_path):
            df_existing = pd.read_csv(save_path)
            print(f"既存の分析結果を読み込みます: {save_path}")
        else:
            df_existing = df.copy()

        # 各分析方法に対して分析実行（既に結果カラムがある場合はスキップ）
        for method in analysis_methods:
            print(f"選択された感情分析方法: {method}")

            # 既に該当する結果が存在するかチェック
            if method == "sentiment" and "Sentiment_Label" in df_existing.columns:
                print("Sentiment結果は既に存在します。スキップします。")
                continue
            if method == "weime" and "Weime_Label" in df_existing.columns:
                print("Weime結果は既に存在します。スキップします。")
                continue
            if method == "mlask" and "MLAsk_Emotion" in df_existing.columns:
                print("MLAsk結果は既に存在します。スキップします。")
                continue

            # 分析実行
            if method == "mlask":
                df_existing["MLAsk_Emotion"] = self.analyze_sentiment(messages, method=method)
            else:
                all_scores = []
                batch_size = 64
                for i in tqdm(range(0, len(messages), batch_size), desc=f"{method} 感情分析中"):
                    batch_messages = messages[i : i + batch_size]
                    all_scores.extend(self.analyze_sentiment(batch_messages, method=method))

                if method == "sentiment":
                    for i, sentiment in enumerate(self.sentiment_names):
                        col = f"Sentiment_{sentiment.capitalize()}"
                        df_existing[col] = [scores[i] for scores in all_scores]

                    sentiment_columns = [f"Sentiment_{sent.capitalize()}" for sent in self.sentiment_names]
                    # 最もスコアが高い列をラベル化
                    df_existing["Sentiment_Label"] = (
                        df_existing[sentiment_columns]
                        .idxmax(axis=1)
                        .str.replace("Sentiment_", "")
                        .str.lower()
                    )

                elif method == "weime":
                    for i, emotion in enumerate(self.emotion_names):
                        col = f"Weime_{emotion}"
                        df_existing[col] = [scores[i] for scores in all_scores]

                    weime_columns = [f"Weime_{emotion}" for emotion in self.emotion_names]
                    # 最もスコアが高い列をラベル化
                    df_existing["Weime_Label"] = (
                        df_existing[weime_columns].idxmax(axis=1).str.replace("Weime_", "")
                    )

            # 途中結果を保存
            df_existing.to_csv(save_path, index=False, encoding="utf-8-sig")
            print(f"{method} の分析結果を保存しました: {save_path}")

        # 最終的なデータフレームをreturn
        return df_existing


if __name__ == "__main__":
    # 実行例
    file_path = "../data/chat_messages/4agZGzQLfF8.csv"
    print(os.path.basename(file_path))

    # クラスをインスタンス化して実行
    analyzer = EmotionAnalyzer(
        checkpoint_sentiment="christian-phu/bert-finetuned-japanese-sentiment",
        checkpoint_weime="../saved_model",
        save_dir="../data/emotion",
    )

    # 複数の感情分析を実行
    df_result = analyzer.analysis_emotion(
        file_path=file_path,
        analysis_methods=["sentiment", "weime", "mlask"]
    )

    print("\n=== 分析結果の先頭5行 ===")
    print(df_result.head())
