# main.py

import os
from datetime import datetime
import pandas as pd

from whisper_comparison import WhisperComparison
from chat_emotions import EmotionAnalyzer
from audio_utils import download_yt_sound, extract_vocals
from chat_download import ChatScraper
from yt_url_utils import YTURLUtils

# インスタンスの初期化
yt_utils = YTURLUtils()


# 動画IDテーブルを取得する関数
def fetch_table_from_db(video_id, db_handler):
    """
    動画IDに対応するテーブルを取得します。
    """
    table_name = f"{video_id}"
    try:
        # テーブルの存在確認
        query_check = f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table_name}'"
        result = pd.read_sql(query_check, db_handler.engine)
        if result.empty:
            print(f"テーブルが存在しません: {table_name}")
            return None

        # テーブルデータの取得
        query_fetch = f"SELECT * FROM {table_name}"
        table_data = pd.read_sql(query_fetch, db_handler.engine)
        return table_data
    except Exception as e:
        print(f"DB取得エラー: {e}")
        return None


if __name__ == "__main__":
    # 入力CSVファイルのパス
    csv_file = "../data/にじさんじ　切り抜き_20250102_202807.csv"
    db_url = "sqlite:///C:/Users/tmkjn/Documents/python/data_analysis/fetch_whisper/chat_data.db"

    audio_dir = "../data/audio"
    os.makedirs(os.path.join(audio_dir, "source"), exist_ok=True)
    os.makedirs(os.path.join(audio_dir, "clipping"), exist_ok=True)

    # ファイル存在確認
    if not os.path.isfile(csv_file):
        print(f"CSVファイル '{csv_file}' が見つかりません。")
        exit()

    # ChatScraperを初期化
    scraper = ChatScraper(db_url=db_url)

    # 入力CSVから元動画URLと切り抜き動画URLを取得
    try:
        input_df = pd.read_csv(csv_file, encoding="utf-8-sig")
    except Exception as e:
        print(f"CSV読み込みエラー: {e}")
        exit()

    search_process_df = scraper.list_original_urls(
        input_df,
        delete_multiple=True,
        url_column="Original videoURL",
        video_url_column="Video URL",
    )
    print(search_process_df.columns)

    # データ保存
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    search_process_df.to_csv(f"../urls_{current_time}.csv", index=False, encoding='utf-8-sig')

    # URLの取得
    source_urls = search_process_df["Original URL"] if "Original URL" in search_process_df.columns else search_process_df["Original videoURL"]
    clipping_urls = search_process_df["Snippet URL"]

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
        source_video_id = yt_utils.get_video_id_from_url(source)
        clip_video_id = yt_utils.get_video_id_from_url(clip)

        # セグメント比較テーブル名を生成
        table_name = f"{clip_video_id}->{source_video_id}"

        # セグメント比較テーブルが既に存在するか確認
        existing_tables = pd.read_sql("SELECT name FROM sqlite_master WHERE type='table'", scraper.db_handler.engine)[
            'name'].values
        segment_comparison_skipped = table_name in existing_tables

        if segment_comparison_skipped:
            print(f"セグメント比較テーブル '{table_name}' は既に存在するためセグメント比較をスキップします。")
        else:
            print(f"セグメント比較を実行します: {table_name}")

        try:
            results.append(
                {"index": i, "source_url": source, "clipping_url": clip, "file_path": None, "status": "処理中"})
            pd.DataFrame(results).to_csv(progress_file, index=False, encoding="utf-8-sig")

            # セグメント比較をスキップしない場合に実行
            if not segment_comparison_skipped:
                # 音声ダウンロード（必要な場合のみ）
                print("元配信音声をダウンロード中...")
                source_audio = download_yt_sound(source, output_dir=os.path.join(audio_dir, "source"))
                print("切り抜き音声をダウンロード中...")
                clipping_audio = download_yt_sound(clip, output_dir=os.path.join(audio_dir, "clipping"))
                clip_path = extract_vocals(clipping_audio)

                source_segments = comparator.transcribe_with_vad(
                    audio_file=source_audio,
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

                # セグメント比較結果を取得
                compare_result = comparator.compare_segments(
                    clipping_segments,
                    source_segments,
                    initial_threshold=0.8,
                    fast_method="sequence",
                    slow_method="tfidf"
                )

                # セグメント比較結果を新しいテーブルとして保存
                compare_result_df = pd.DataFrame(compare_result)
                compare_result_df.to_sql(
                    table_name,
                    con=scraper.db_handler.engine,
                    if_exists='replace',
                    index=False
                )
                print(f"セグメント比較結果をテーブル '{table_name}' として保存しました。")

            source_table = fetch_table_from_db(source_video_id, scraper.db_handler)
            if source_table is None or source_table.empty:
                print(f"動画ID {source_video_id} のデータが見つかりません。感情分析をスキップします。")
                continue

            analysis_result = emotion_comparator.analysis_emotion(
                df=source_table,  # 必要に応じて正しい形式に変換
                analysis_methods=["sentiment", "weime", "mlask"]
            )
            print(f"感情分析が完了しました: {analysis_result}")

            # 感情分析結果をDBに保存
            scraper.db_handler.save_emotion_analysis(source_video_id, analysis_result)

            results[-1]["status"] = "成功"
            pd.DataFrame(results).to_csv(progress_file, index=False, encoding="utf-8-sig")

        except Exception as e:
            results[-1]["status"] = f"失敗: {str(e)}"
            pd.DataFrame(results).to_csv(progress_file, index=False, encoding="utf-8-sig")
            print(f"エラーが発生しました: {str(e)}")

    # 全処理完了
    print("\n全ての動画処理が完了しました。")
    results_df = pd.DataFrame(results)
    results_df.to_csv(f"processing_{datetime.now().strftime('%Y%m%d_%H%M%S')}_results.csv", index=False,
                      encoding="utf-8-sig")
    print("処理結果を保存しました。")
