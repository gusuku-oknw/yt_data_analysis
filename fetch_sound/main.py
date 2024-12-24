from datetime import datetime
import os
import pandas as pd
import logging
from chat_utils import list_original_urls
from chat_processor import ChatProcessor
from AudioComparator import AudioComparator
from chat_emotions import EmotionAnalyzer
from yt_dlp import YoutubeDL
import torchaudio

# YouTube動画または音声をダウンロード
def download_yt_sound(url, output_dir="../data/sound", audio_only=True):
    """
    YouTube動画または音声をダウンロード。

    Parameters:
        url (str): ダウンロード対象のURL。
        output_dir (str): 出力ディレクトリ。
        audio_only (bool): Trueの場合、音声のみをダウンロード。

    Returns:
        str: ダウンロードされたファイルのパス。
    """
    os.makedirs(output_dir, exist_ok=True)

    # 出力ファイル名を生成（拡張子なし）
    video_id = ChatProcessor.get_video_id_from_url(ChatProcessor.remove_query_params(url))
    file_name = video_id  # 拡張子はFFmpegで付加
    file_path_no_ext = os.path.join(output_dir, file_name)
    file_path_with_ext = f"{file_path_no_ext}.mp3"

    # ファイルが存在する場合はダウンロードをスキップ
    if os.path.exists(file_path_with_ext):
        logging.info(f"ファイルが既に存在します。スキップします: {file_path_with_ext}")
        return file_path_with_ext

    ydl_opts = {
        'format': 'bestaudio' if audio_only else 'bestvideo+bestaudio',
        'outtmpl': file_path_no_ext,  # 拡張子を付けない
        'noplaylist': True,
        'quiet': False,
        'postprocessors': [  # 音声のみの場合の後処理
            {
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': '192',
            }
        ] if audio_only else None
    }

    # YoutubeDLを使用してダウンロード
    with YoutubeDL(ydl_opts) as ydl:
        ydl.extract_info(url, download=True)

    print(f"ファイルをダウンロードしました: {file_path_with_ext}")
    return file_path_with_ext

def download_and_transcribe(source, clip):
    """
    ダウンロードと文字起こしの処理を実行。
    """
    try:
        logging.info(f"Starting download and transcribe for Source: {source}, Clip: {clip}")
        source_path = download_yt_sound(source, output_dir="../data/audio/source")
        clip_path = download_yt_sound(clip, output_dir="../data/audio/clipping")
        logging.info(f"Downloaded Source: {source_path}, Clip: {clip_path}")
        comparator = AudioComparator(sampling_rate=torchaudio.info(clip_path).sample_rate)

        result = comparator.process_audio(source_path, clip_path)
        logging.info(f"Download and transcribe completed for Source: {source}, Clip: {clip}")
        return {"status": "success", "result": result}
    except Exception as e:
        logging.error(f"Error during download and transcribe: {e}")
        return {"status": "error", "error": str(e)}

def main_emotion_analysis(file_path, analysis_methods, plot_results, plot_save, save_dir):
    """
    感情分析のメイン処理。
    """
    analyzer = EmotionAnalyzer(output_dir=save_dir)
    return analyzer.analyze_file(
        file_path=file_path,
        analysis_methods=analysis_methods,
        plot_results=plot_results,
        plot_save_path=plot_save
    )

def main():
    # ログ設定
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # ディレクトリ構成の設定
    data_dir = "../data"
    audio_dir = os.path.join(data_dir, "audio")
    transcription_dir = os.path.join(data_dir, "transcription")
    comparison_dir = os.path.join(data_dir, "comparison")

    # 各ディレクトリの作成
    os.makedirs(os.path.join(audio_dir, "source"), exist_ok=True)
    os.makedirs(os.path.join(audio_dir, "clipping"), exist_ok=True)
    os.makedirs(os.path.join(transcription_dir, "source"), exist_ok=True)
    os.makedirs(os.path.join(transcription_dir, "clipping"), exist_ok=True)
    os.makedirs(os.path.join(comparison_dir, "matches"), exist_ok=True)
    os.makedirs(os.path.join(comparison_dir, "unmatched"), exist_ok=True)

    # 現在時刻を取得し、フォーマット
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    csv_search_filename = f"urls_{current_time}.csv"
    csv_filename = "../data/test_videos_processed.csv"

    # チャットデータの処理
    logging.info("チャットデータのURLリストを取得中...")
    search_process_df = list_original_urls(csv_filename)
    logging.info(f"取得したデータフレーム: \n{search_process_df}")
    search_process_df.to_csv(csv_search_filename, index=False, encoding="utf-8-sig")

    source_url = search_process_df["Original URL"]
    clipping_url = search_process_df["Video URL"]
    source_file = search_process_df.get("File Path", [])

    results = []
    progress_file = "../processing_progress.csv"

    if not os.path.exists(progress_file):
        pd.DataFrame(columns=["index", "source_url", "clipping_url", "file_path", "status"]).to_csv(
            progress_file, index=False, encoding="utf-8-sig"
        )

    for i, (source, clip, file_path) in enumerate(zip(source_url, clipping_url, source_file), start=1):
        matches_filename = (f"../data/matches/"
                            f"{ChatProcessor.get_video_id_from_url(ChatProcessor.remove_query_params(source))}_"
                            f"{ChatProcessor.get_video_id_from_url(ChatProcessor.remove_query_params(clip))}.csv")

        logging.info(f"Processing {i}/{len(source_url)}: Source={source}, Clip={clip}")
        logging.debug(f"Generated matches_filename: {matches_filename}")

        if os.path.exists(matches_filename):
            logging.info(f"File already exists: {matches_filename}. Skipping.")
            continue

        results.append(
            {"index": i, "source_url": source, "clipping_url": clip, "file_path": file_path, "status": "文字起こし中"}
        )
        pd.DataFrame(results).to_csv(progress_file, index=False, encoding="utf-8-sig")

        try:
            result = download_and_transcribe(source, clip)
            results[-1]["status"] = result["status"] if result["status"] == "success" else f"失敗: {result.get('error', '詳細不明')}"
            pd.DataFrame(results).to_csv(progress_file, index=False, encoding="utf-8-sig")

            if result["status"] == "success":
                # 結果をCSVに保存
                if result:
                    df = pd.DataFrame(result["blocks"])
                    df.to_csv(matches_filename, index=False, encoding='utf-8-sig')
                    logging.info(f"文字起こし結果が {matches_filename} に保存されました。")

                    print("Transcriptions:")
                    for block in result:
                        print(f"Clip Start: {block['clip_start']}, Clip End: {block['clip_end']}, "
                              f"Source Start: {block['source_start']}, Source End: {block['source_end']}, "
                              f"Matched: {block['matched']}, Distance: {block['distance']}, "
                              f"Text: \"{block['text']}\", 分散: {block['variance']}, 閾値: {block['threshold']}")
                else:
                    logging.info("マッチするブロックが見つからないか、文字起こしに失敗しました。")

                logging.info(f"\n最終的な閾値: {result['result']['current_threshold']}")

                results[-1]["status"] = "感情分析中"
                pd.DataFrame(results).to_csv(progress_file, index=False, encoding="utf-8-sig")

                analysis_result = main_emotion_analysis(
                    file_path=file_path,
                    analysis_methods=["sentiment", "weime", "mlask"],
                    plot_results=False,
                    plot_save=None,
                    save_dir="../data/emotion"
                )

                results[-1]["status"] = "完了"
                pd.DataFrame(results).to_csv(progress_file, index=False, encoding="utf-8-sig")

        except Exception as e:
            logging.error(f"Error processing video {i}: {e}")

    results_df = pd.DataFrame(results)
    results_df.to_csv(f"processing_{current_time}_results.csv", index=False, encoding="utf-8-sig")
    logging.info("全ての処理が完了しました。")

if __name__ == "__main__":
    main()
