import os
import logging
import traceback
from datetime import datetime
import pandas as pd
from sqlalchemy import inspect, text

from whisper_comparison import WhisperComparison
from chat_emotions import EmotionAnalyzer
from audio_utils import download_yt_sound, extract_vocals
from chat_download import ChatScraper
from yt_url_utils import YTURLUtils

# インスタンスの初期化
yt_utils = YTURLUtils()

# ログの設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# テーブルデータを取得

def fetch_table_from_db(video_id, db_handler):
    """動画IDに対応するテーブルを取得します。"""
    table_name = f"{video_id}"
    try:
        # テーブル名をダブルクォートで囲む
        query_check = f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table_name}'"
        if pd.read_sql(query_check, db_handler.engine).empty:
            logging.warning(f"テーブルが存在しません: {table_name}")
            return None

        # テーブル名をダブルクォートで囲む
        query_fetch = f"SELECT * FROM \"{table_name}\""
        return pd.read_sql(query_fetch, db_handler.engine)
    except Exception as e:
        logging.error(f"DB取得エラー: {e}")
        logging.debug(traceback.format_exc())
        return None



def add_missing_columns(engine, table_name, required_columns):
    """必要なカラムをデータベーステーブルに追加します。"""
    inspector = inspect(engine)
    existing_columns = {col['name'].lower() for col in inspector.get_columns(table_name)}

    for column_name, column_type in required_columns.items():
        if column_name.lower() not in existing_columns:
            try:
                with engine.connect() as connection:
                    # テーブル名をエスケープ
                    escaped_table_name = f'"{table_name}"'
                    connection.execute(text(f"ALTER TABLE {escaped_table_name} ADD COLUMN {column_name} {column_type.upper()}"))
                logging.info(f"カラム '{column_name}' を追加しました。")
            except Exception as e:
                logging.error(f"カラム追加時にエラーが発生しました: {e}")
                logging.debug(traceback.format_exc())
        else:
            logging.info(f"カラム '{column_name}' は既に存在しています。")


def initialize_directories(base_dir):
    """必要なディレクトリを作成します。"""
    os.makedirs(os.path.join(base_dir, "source"), exist_ok=True)
    os.makedirs(os.path.join(base_dir, "clipping"), exist_ok=True)


def process_video(source, clip, comparator, emotion_analyzer, scraper, audio_dir, progress_file):
    """動画の処理を実行します。"""
    source_video_id = yt_utils.get_video_id_from_url(source)
    clip_video_id = yt_utils.get_video_id_from_url(clip)
    table_name = f"{clip_video_id}->{source_video_id}"

    # セグメント比較テーブルの存在確認
    existing_tables = pd.read_sql(
        "SELECT name FROM sqlite_master WHERE type='table'",
        scraper.db_handler.engine
    )['name'].values

    try:
        if table_name in existing_tables:
            logging.info(f"セグメント比較テーブル '{table_name}' は既に存在しています。")
        else:
            # 音声のダウンロード
            source_audio = download_yt_sound(source, output_dir=os.path.join(audio_dir, "source"))
            clipping_audio = download_yt_sound(clip, output_dir=os.path.join(audio_dir, "clipping"))
            clip_path = extract_vocals(clipping_audio)

            # 音声のテキスト変換
            source_segments = comparator.transcribe_with_vad(
                audio_file=source_audio, threshold=0.35, language="ja", beam_size=5
            )
            clipping_segments = comparator.transcribe_with_vad(
                audio_file=clip_path, threshold=0.35, language="ja", beam_size=5
            )

            # セグメント比較
            compare_result = comparator.compare_segments(
                clipping_segments,
                source_segments,
                initial_threshold=0.8,
                fast_method="sequence",
                slow_method="tfidf"
            )

            # 結果を保存
            pd.DataFrame(compare_result).to_sql(
                table_name, con=scraper.db_handler.engine, if_exists='replace', index=False
            )
            logging.info(f"セグメント比較結果をテーブル '{table_name}' に保存しました。")

        # 感情分析
        source_table = fetch_table_from_db(source_video_id, scraper.db_handler)
        if source_table is not None and not source_table.empty:
            required_columns = {
                "sentiment_positive": "FLOAT",
                "sentiment_neutral": "FLOAT",
                "sentiment_negative": "FLOAT",
                "sentiment_label": "TEXT",
                "weime_joy": "FLOAT",
                "weime_sadness": "FLOAT",
                "weime_anticipation": "FLOAT",
                "weime_surprise": "FLOAT",
                "weime_anger": "FLOAT",
                "weime_fear": "FLOAT",
                "weime_disgust": "FLOAT",
                "weime_trust": "FLOAT",
                "weime_label": "TEXT",
                "mlask_emotion": "TEXT",
            }
            add_missing_columns(scraper.db_handler.engine, source_video_id, required_columns)

            analysis_result = emotion_analyzer.analysis_emotion(
                df=source_table,
                analysis_methods=["sentiment", "weime", "mlask"]
            )
            analysis_result.to_sql(
                source_video_id, con=scraper.db_handler.engine, if_exists='replace', index=False, chunksize=1000
            )
            logging.info("感情分析結果を保存しました。")

    except Exception as e:
        logging.error(f"動画処理中にエラーが発生しました: {e}")
        logging.debug(traceback.format_exc())


if __name__ == "__main__":
    csv_file = "../data/にじさんじ　切り抜き_20250102_202807.csv"
    db_url = "sqlite:///C:/Users/tmkjn/Documents/python/data_analysis/fetch_whisper/chat_data.db"
    audio_dir = "../data/audio"
    progress_file = "../processing_progress.csv"

    initialize_directories(audio_dir)

    if not os.path.isfile(csv_file):
        logging.error(f"CSVファイル '{csv_file}' が見つかりません。")
        exit()

    scraper = ChatScraper(db_url=db_url)
    comparator = WhisperComparison(sampling_rate=16000)
    emotion_analyzer = EmotionAnalyzer()

    try:
        input_df = pd.read_csv(csv_file, encoding="utf-8-sig")
    except Exception as e:
        logging.error(f"CSV読み込みエラー: {e}")
        exit()

    search_process_df = scraper.list_original_urls(
        input_df,
        delete_multiple=True,
        url_column="Original videoURL",
        video_url_column="Video URL",
    )
    search_process_df.to_csv(f"../urls_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.csv", index=False, encoding='utf-8-sig')

    source_urls = search_process_df.get("Original URL", search_process_df.get("Original videoURL"))
    clipping_urls = search_process_df["Snippet URL"]

    for i, (source, clip) in enumerate(zip(source_urls, clipping_urls), start=1):
        logging.info(f"\n=== {i}/{len(source_urls)} 番目の動画の処理を開始します ===")
        process_video(source, clip, comparator, emotion_analyzer, scraper, audio_dir, progress_file)

    logging.info("全ての動画処理が完了しました。")
