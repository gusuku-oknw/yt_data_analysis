import os
import pandas as pd
import re
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F
from tqdm import tqdm
from mlask import MLAsk
import logging

# ロギングの設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


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
        logging.info(f"使用デバイス: {self.device}")

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
        df = df[df["time_in_seconds"] >= 0].copy()

        # 時間を整数秒に丸める
        df["time_in_seconds"] = df["time_in_seconds"].round().astype(int)

        # メッセージのクリーンアップ関数
        def clean_message(x):
            x = re.sub(r":_[A-Z]+:", "", str(x))
            x = re.sub(r"[^\w\s]", "", x)
            return x.strip()

        df["message"] = df["message"].apply(clean_message)
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
    def analysis_emotion(self, df, analysis_methods=["sentiment"]):
        # 全てのカラム名を小文字に変換
        df.columns = df.columns.str.lower()

        # カラム確認
        if "time_in_seconds" not in df.columns or df["time_in_seconds"].dropna().empty:
            logging.error("'time_in_seconds' カラムが存在しないか、データが空です。")
            return df

        if "message" not in df.columns or df["message"].dropna().empty:
            logging.error("'message' カラムが存在しないか、データが空です。")
            return df

        # データ読み込みと前処理
        df = self.preprocess(df)
        messages = df["message"].tolist()
        df_existing = df.copy()

        # 各分析方法に対して分析実行
        for method in analysis_methods:
            logging.info(f"選択された感情分析方法: {method}")

            # スキップ条件
            if method == "sentiment" and not any(
                    col.startswith("sentiment_") for col in df_existing.columns):
                logging.info("Sentimentの結果が欠けています。分析を実行します。")
            elif method == "weime" and not any(col.startswith("weime_") for col in df_existing.columns):
                logging.info("Weimeの結果が欠けています。分析を実行します。")
            elif method == "mlask" and "mlask_emotion" not in df_existing.columns:
                logging.info("MLAskの結果が欠けています。分析を実行します。")
            else:
                logging.info(f"{method} の結果が既に存在します。スキップします。")
                continue

            # 分析実行
            if method == "mlask":
                logging.info("MLAsk 感情分析を実行中...")
                df_existing["mlask_emotion"] = self.analyze_sentiment(messages, method=method)
            else:
                all_scores = []
                batch_size = 64
                for i in tqdm(range(0, len(messages), batch_size), desc=f"{method} 感情分析中"):
                    batch_messages = messages[i: i + batch_size]
                    analyzed_scores = self.analyze_sentiment(batch_messages, method=method)

                    if not analyzed_scores or all(score is None for score in analyzed_scores):
                        logging.error(f"{method} の分析結果が無効です: {analyzed_scores}")
                        continue

                    all_scores.extend(analyzed_scores)

                if method == "sentiment":
                    for i, sentiment in enumerate(self.sentiment_names):
                        col = f"sentiment_{sentiment.lower()}"  # 小文字に統一
                        df_existing[col] = [scores[i] for scores in all_scores]

                    sentiment_columns = [f"sentiment_{sent.lower()}" for sent in self.sentiment_names]
                    df_existing["sentiment_label"] = (
                        df_existing[sentiment_columns]
                        .idxmax(axis=1)
                        .str.replace("sentiment_", "")
                        .str.lower()
                    )

                elif method == "weime":
                    for i, emotion in enumerate(self.emotion_names):
                        col = f"weime_{emotion.lower()}"  # 小文字に統一
                        df_existing[col] = [scores[i] for scores in all_scores]

                    weime_columns = [f"weime_{emotion.lower()}" for emotion in self.emotion_names]
                    df_existing["weime_label"] = (
                        df_existing[weime_columns].idxmax(axis=1).str.replace("weime_", "")
                    )

        # 全てのカラム名を小文字に統一（既に小文字に統一済みの場合は不要）
        df_existing.columns = [col.lower() for col in df_existing.columns]

        logging.info("感情分析が完了しました。データフレームを返します。")
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
