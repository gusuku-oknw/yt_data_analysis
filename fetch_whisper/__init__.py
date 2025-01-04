import os
from datetime import datetime
import pandas as pd

from whisper_comparison import WhisperComparison
from chat_emotions import EmotionAnalyzer
from audio_utils import download_yt_sound, extract_vocals
from chat_download import ChatScraper
from yt_url_utils import YTURLUtils
yt_utils = YTURLUtils()

if __name__ == "__main__":
    # 入力CSVファイルのパス
    csv_file = "../data/にじさんじ　切り抜き_20250102_202807.csv"
    db_url = "sqlite:///C:/Users/tmkjn/Documents/python/data_analysis/fetch_whisper/chat_data.db"

    try:
        os.path.isfile(csv_file)
    except FileNotFoundError:
        print(f"CSVファイル '{csv_file}' が見つかりません。")
        exit()

    # ChatScraperを初期化
    scraper = ChatScraper(db_url=db_url)

    # 入力CSVから元動画URLと切り抜き動画URLを取得
    search_process_df = scraper.list_original_urls(
        pd.read_csv(csv_file, encoding="utf-8-sig"),
        delete_multiple=True,
        url_column="Original videoURL",
        video_url_column="Video URL",
    )
    print(search_process_df.columns)

    # データ保存
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    search_process_df.to_csv(f"../urls_{current_time}.csv", index=False, encoding='utf-8-sig')

    # URLやパスの取得
    source_urls = search_process_df["Original URL"] if "Original URL" in search_process_df.columns else search_process_df["Original videoURL"]
    clipping_urls = search_process_df["Video URL"]

    print(f"元配信URL: {source_urls.iloc[0]}")
    print(f"切り抜きURL: {clipping_urls.iloc[0]}")

    # DBからファイルパスを取得する処理
    def fetch_file_path_from_db(video_id, db_handler):
        query = f"SELECT file_path FROM Channel_videos WHERE video_id = '{video_id}'"
        try:
            result = pd.read_sql(query, db_handler.engine)
            return result['file_path'].iloc[0] if not result.empty else None
        except Exception as e:
            print(f"DB取得エラー: {e}")
            return None

    # 処理進捗を管理するファイルの準備
    progress_file = "../processing_progress.csv"
    if not os.path.exists(progress_file):
        pd.DataFrame(columns=["index", "source_url", "clipping_url", "file_path", "status"]).to_csv(
            progress_file, index=False, encoding="utf-8-sig"
        )

    # WhisperComparatorとEmotionAnalyzerの初期化
    comparator = WhisperComparison(sampling_rate=16000)
    emotion_comparator = EmotionAnalyzer()

    # 処理結果を保存するリスト
    results = []

    for i, (source, clip) in enumerate(zip(source_urls, clipping_urls), start=1):
        print(f"\n=== {i}/{len(source_urls)} 番目の動画の処理を開始します ===")
        matches_filename = f"../data/matches/{yt_utils.get_video_id_from_url(yt_utils.remove_query_params(source))}_{yt_utils.get_video_id_from_url(yt_utils.remove_query_params(clip))}.csv"

        if os.path.exists(matches_filename):
            print(f"File already exists: {matches_filename}. Skipping.")
            continue

        # DBからファイルパスを取得
        source_file_path = fetch_file_path_from_db(yt_utils.get_video_id_from_url(source), scraper.db_handler)
        clipping_file_path = fetch_file_path_from_db(yt_utils.get_video_id_from_url(clip), scraper.db_handler)

        if not source_file_path or not clipping_file_path:
            print(f"ファイルパスが取得できませんでした: source={source_file_path}, clip={clipping_file_path}")
            results.append({"index": i, "source_url": source, "clipping_url": clip, "file_path": None, "status": "ファイルパス取得失敗"})
            pd.DataFrame(results).to_csv(progress_file, index=False, encoding="utf-8-sig")
            continue

        try:
            # 処理進捗の更新
            results.append({"index": i, "source_url": source, "clipping_url": clip, "file_path": source_file_path, "status": "文字起こし中"})
            pd.DataFrame(results).to_csv(progress_file, index=False, encoding="utf-8-sig")

            # 音声処理
            clip_path = extract_vocals(clipping_file_path)

            # 元動画と切り抜き動画のセグメント化と文字起こし
            source_segments = comparator.transcribe_with_vad(
                audio_file=source_file_path,
                threshold=0.35,
                language="ja",
                beam_size=5
            )
            clipping_segments = comparator.transcribe_with_vad(
                audio_file=clip_path,
                threshold=0.35,
                language="ja",
                beam_size=5
            )

            # セグメント比較
            matches, unmatched_source = comparator.compare_segments(
                clipping_segments,
                source_segments,
                initial_threshold=0.8,
                fast_method="sequence",
                slow_method="tfidf"
            )

            # 感情分析
            analysis_result = emotion_comparator.analysis_emotion(
                file_path=source_file_path,
                analysis_methods=["sentiment", "weime", "mlask"],
            )
            print(f"感情分析が完了しました: {analysis_result}")

            # 成功時の処理
            results[-1]["status"] = "成功"
            pd.DataFrame(results).to_csv(progress_file, index=False, encoding="utf-8-sig")

        except Exception as e:
            # 例外発生時の処理
            results[-1]["status"] = f"失敗: {str(e)}"
            pd.DataFrame(results).to_csv(progress_file, index=False, encoding="utf-8-sig")
            print(f"エラーが発生しました: {str(e)}")

    # 全処理完了
    print("\n全ての動画処理が完了しました。")
    results_df = pd.DataFrame(results)
    results_df.to_csv(f"processing_{datetime.now().strftime('%Y%m%d_%H%M%S')}_results.csv", index=False, encoding="utf-8-sig")
    print("処理結果を保存しました。")