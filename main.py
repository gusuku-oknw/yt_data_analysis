from datetime import datetime
import os
from tqdm import tqdm

import pandas as pd
import data_collection
import search_csv_chat_download
from audio_transcription_comparator import download_yt_sound
from audio_transcription_comparator import audio_transcription2csv
from audio_transcription_comparator import compare_segments
from analyze_chat_emotions import main_emotion_analysis
from chat_download import get_video_id_from_url, remove_query_params, chat_download_csv
from search_csv_chat_download import list_original_urls
from data_collection import search_main, get_popular_main

def download_and_transcribe(source_url, clipping_url):
    """
    元配信音声と切り抜き音声を比較して一致セグメントと不一致セグメントを保存する関数。

    Parameters:
        source_url (str): 元配信音声のYouTube URL。
        clipping_url (str): 切り抜き音声のYouTube URL。

    Returns:
        dict: 処理結果のステータスとファイルパス。
    """
    try:
        # ディレクトリ構成の設定
        data_dir = "data"
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

        # ステップ1: 音声ダウンロード
        print("元配信音声をダウンロード中...")
        source_audio = download_yt_sound(source_url, output_dir=os.path.join(audio_dir, "source"))
        print("切り抜き音声をダウンロード中...")
        clipping_audio = download_yt_sound(clipping_url, output_dir=os.path.join(audio_dir, "clipping"))

        # ステップ2: Distil-Whisperで文字起こし
        print("元配信音声を文字起こし中...")
        source_silences = audio_transcription2csv(
            source_audio,
            output_directory=os.path.join(transcription_dir, "source"),
            extract=False
        )
        print("切り抜き音声を文字起こし中...")
        clipping_silences = audio_transcription2csv(
            clipping_audio,
            output_directory=os.path.join(transcription_dir, "clipping")
        )

        # ステップ3: テキストの比較
        print(len(source_silences), len(clipping_silences))
        matches, unmatched = compare_segments(
            clipping_silences,
            source_silences,
            initial_threshold=0.8,
            fast_method="sequence",  # まずは高速なsequenceでマッチング
            slow_method="tfidf"  # Unmatchedに対してのみtfidfで再マッチング
        )

        # ファイル名を作成
        source_basename = os.path.basename(source_audio).replace(".mp3", "")
        clipping_basename = os.path.basename(clipping_audio).replace(".mp3", "")
        match_file = os.path.join(comparison_dir, "matches", f"{source_basename}_{clipping_basename}_matches.csv")
        unmatched_file = os.path.join(comparison_dir, "unmatched", f"{source_basename}_{clipping_basename}_unmatched.csv")

        # CSVに保存
        pd.DataFrame(matches).to_csv(match_file, index=False, encoding="utf-8-sig")
        print(f"一致したセグメントを保存しました: {match_file}")

        if unmatched:
            pd.DataFrame(unmatched).to_csv(unmatched_file, index=False, encoding="utf-8-sig")
            print(f"一致しなかったセグメントを保存しました: {unmatched_file}")

        print("処理が完了しました。")

        return {
            "status": "success",
            "match_file": match_file,
            "unmatched_file": unmatched_file
        }

    except Exception as e:
        print(f"処理中にエラーが発生しました: {e}")
        return {
            "status": "failed",
            "error": str(e)
        }


if __name__ == "__main__":
    # search_keyword = "博衣こより　切り抜き"
    # 現在時刻を取得し、フォーマットする
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # csv_filename = f"{search_keyword}_{current_time}_videos.csv"

    # チャットデータのダウンロード
    # search_main(search_keyword = "博衣こより　切り抜き")
    # search_descriptions_df = data_collection.search_main(search_keyword)
    # search_process_df = data_collection.add_original_video_urls(search_descriptions_df)
    # search_process_df.to_csv(csv_filename, index=False, encoding='utf-8-sig')
    # get_popular_main(channel_id = "UCs6nmQViDpUw0nuIx9c_WvA")

    # チャットデータのダウンロード
    csv_search_filename = f"urls_{current_time}.csv"
    csv_filename = "data/兎田ぺこら　切り抜き_2024-11-16_20-48-27_videos_processed.csv"
    search_process_df = search_csv_chat_download.list_original_urls(csv_filename)
    search_process_df.to_csv(csv_search_filename, index=False, encoding='utf-8-sig')

    # 元配信URLと切り抜きURL
    source_url = search_process_df["Original URL"]
    clipping_url = search_process_df["Video URL"]
    source_file = search_process_df["File Path"]
    print(f"元配信URL: {source_url}")
    print(f"切り抜きURL: {clipping_url}")

    # 結果を記録するリスト
    results = []
    analysis_files = []
    progress_file = "processing_progress.csv"

    # 初期化
    if not os.path.exists(progress_file):
        pd.DataFrame(columns=["index", "source_url", "clipping_url", "file_path", "status"]).to_csv(
            progress_file, index=False, encoding="utf-8-sig"
        )

    for i, (source, clip, file_path) in enumerate(zip(source_url, clipping_url, source_file), start=1):
        print(f"\n=== {i} 番目の動画の処理を開始します ===")
        print(f"元配信URL: {source}")
        print(f"切り抜きURL: {clip}")
        matches_filename = f"data/matches/{get_video_id_from_url(remove_query_params(source))}_{get_video_id_from_url(remove_query_params(clip))}.csv"
        print(f"matches_filename: {matches_filename}")
        # すでに処理済みの場合はスキップ# ファイルが存在するか確認
        if os.path.exists(matches_filename):
            print(f"File already exists: {matches_filename}. Skipping.")
            continue
        # ステータス更新: 文字起こし中
        results.append(
            {"index": i, "source_url": source, "clipping_url": clip, "file_path": file_path, "status": "文字起こし中"})
        pd.DataFrame(results).to_csv(progress_file, index=False, encoding="utf-8-sig")

        # ダウンロードと文字起こしの処理
        result = download_and_transcribe(source, clip)
        results[-1]["status"] = result["status"] if result[
                                                        "status"] == "success" else f"失敗: {result.get('error', '詳細不明')}"
        pd.DataFrame(results).to_csv(progress_file, index=False, encoding="utf-8-sig")

        if result["status"] == "success":
            print(f"ダウンロードと文字起こしが成功しました: {file_path}")

            # ステータス更新: 感情分析中
            results[-1]["status"] = "感情分析中"
            pd.DataFrame(results).to_csv(progress_file, index=False, encoding="utf-8-sig")

            # 感情分析の実行
            analysis_result = main_emotion_analysis(
                file_path=file_path,
                analysis_methods=["sentiment", "weime", "mlask"],
                plot_results=False,  # プロットを表示しない
                plot_save=None,
                # プロット画像を保存
                save_dir="data/emotion"
            )
            analysis_files.append(analysis_result)
            print(f"{i} 番目の動画の感情分析が完了しました。")

            # ステータス更新: 完了
            results[-1]["status"] = "完了"
            pd.DataFrame(results).to_csv(progress_file, index=False, encoding="utf-8-sig")
        else:
            print(f"{i} 番目の動画の処理が失敗しました: {result.get('error', '詳細不明のエラー')}")

    print("\n全ての動画処理が完了しました。")

    # 結果をデータフレームに保存
    results_df = pd.DataFrame(results)
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_df.to_csv(f"processing_{current_time}_results.csv", index=False, encoding="utf-8-sig")
    print(f"処理結果を 'processing_{current_time}_results.csv' に保存しました。")
