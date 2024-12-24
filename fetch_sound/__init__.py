from datetime import datetime
import os
import pandas as pd
from chat_utils import list_original_urls
from chat_processor import ChatProcessor, ChatDownloader
from chat_emotions import EmotionAnalyzer


class VideoProcessingPipeline:
    def __init__(self, base_dir="../data", progress_file="processing_progress.csv"):
        """
        動画処理パイプラインの初期化。

        Parameters:
            base_dir (str): データ保存の基本ディレクトリ。
            progress_file (str): 処理進捗を記録するファイル名。
        """
        self.base_dir = base_dir
        self.progress_file = os.path.join(base_dir, progress_file)
        self.matches_dir = os.path.join(base_dir, "matches")
        self.emotion_dir = os.path.join(base_dir, "emotion")
        self.initialize_directories()

    def initialize_directories(self):
        """
        必要なディレクトリ構造を作成。
        """
        os.makedirs(self.base_dir, exist_ok=True)
        os.makedirs(self.matches_dir, exist_ok=True)
        os.makedirs(self.emotion_dir, exist_ok=True)

    def process_videos(self, input_csv, output_csv=None):
        """
        動画のダウンロード、文字起こし、感情分析を順次実行。

        Parameters:
            input_csv (str): 入力CSVファイルのパス。
            output_csv (str): 処理結果を保存するCSVファイルパス (省略時は自動生成)。
        """
        current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        if not output_csv:
            output_csv = os.path.join(self.base_dir, f"processing_{current_time}_results.csv")

        # 入力CSVからデータを取得
        video_data = list_original_urls(input_csv, base_directory=self.base_dir)
        video_data.to_csv(output_csv, index=False, encoding="utf-8-sig")

        # 処理の進捗記録用の初期化
        if not os.path.exists(self.progress_file):
            pd.DataFrame(columns=["index", "source_url", "clipping_url", "file_path", "status"]).to_csv(
                self.progress_file, index=False, encoding="utf-8-sig"
            )

        results = []

        # 元動画と切り抜き動画の処理を開始
        for i, row in video_data.iterrows():
            source_url = row["Original URL"]
            clipping_url = row["Video URL"]
            file_path = row.get("File Path", "")

            print(f"\n=== {i + 1}/{len(video_data)} 番目の動画を処理中 ===")
            matches_filename = os.path.join(
                self.matches_dir,
                f"{get_video_id_from_url(remove_query_params(source_url))}_{get_video_id_from_url(remove_query_params(clipping_url))}.csv"
            )

            # 処理済みの場合はスキップ
            if os.path.exists(matches_filename):
                print(f"既に処理済み: {matches_filename}")
                continue

            # ダウンロードと文字起こし
            results.append({"index": i + 1, "source_url": source_url, "clipping_url": clipping_url, "file_path": file_path, "status": "文字起こし中"})
            self.update_progress(results)

            result = download_and_transcribe(source_url, clipping_url)
            results[-1]["status"] = result["status"] if result["status"] == "success" else f"失敗: {result.get('error', '詳細不明')}"
            self.update_progress(results)

            if result["status"] == "success":
                # 感情分析の実行
                print("感情分析を実行中...")
                results[-1]["status"] = "感情分析中"
                self.update_progress(results)

                analysis_result = main_emotion_analysis(
                    file_path=file_path,
                    analysis_methods=["sentiment", "weime", "mlask"],
                    plot_results=False,
                    save_dir=self.emotion_dir
                )
                results[-1]["status"] = "完了"
                self.update_progress(results)

        # 処理結果を保存
        results_df = pd.DataFrame(results)
        results_df.to_csv(output_csv, index=False, encoding="utf-8-sig")
        print(f"全ての処理が完了しました: {output_csv}")

    def update_progress(self, results):
        """
        処理進捗を記録。
        """
        pd.DataFrame(results).to_csv(self.progress_file, index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    pipeline = VideoProcessingPipeline()
    pipeline.process_videos(input_csv="../data/test_processed.csv")
