from search_csv_chat_download import list_original_urls
from datetime import datetime
import os
import pandas as pd
from analyze_chat_emotions import main_emotion_analysis
from chat_download import get_video_id_from_url, remove_query_params
from audio_transcription_comparator import download_yt_sound, audio_transcription2csv, compare_segments
from fetch_yt_source_and_clip import download_and_transcribe


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
    csv_search_filename = f"../urls_{current_time}.csv"
    csv_filename = "../data/葛葉　切り抜き_2024-11-16_20-48-12_videos_processed.csv"
    search_process_df = list_original_urls(csv_filename)
    search_process_df.to_csv(csv_search_filename, index=False, encoding='utf-8-sig')

    # 元配信URLと切り抜きURL
    source_url = search_process_df["Original URL"]
    clipping_url = search_process_df["Video URL"]
    source_file = search_process_df["File Path"]
    print(f"元配信URL:\n {source_url}")
    print(f"切り抜きURL:\n {clipping_url}")

    # 結果を記録するリスト
    results = []
    analysis_files = []
    progress_file = "../processing_progress.csv"

    # 初期化
    if not os.path.exists(progress_file):
        pd.DataFrame(columns=["index", "source_url", "clipping_url", "file_path", "status"]).to_csv(
            progress_file, index=False, encoding="utf-8-sig"
        )

    for i, (source, clip, file_path) in enumerate(zip(source_url, clipping_url, source_file), start=1):
        print(f"\n=== {i}/{len(source_url)} 番目の動画の処理を開始します ===")
        print(f"元配信URL: {source}")
        print(f"切り抜きURL: {clip}")
        matches_filename = f"../data/matches/{get_video_id_from_url(remove_query_params(source))}_{get_video_id_from_url(remove_query_params(clip))}.csv"
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
                save_dir="../data/emotion"
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
