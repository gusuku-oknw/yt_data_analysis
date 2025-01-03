import os
from datetime import datetime
import pandas as pd

from chat_download import list_original_urls, get_video_id_from_url, remove_query_params
from whisper_comparison import WhisperComparison
from chat_emotions import EmotionAnalyzer
from audio_utils import download_yt_sound, extract_vocals


if __name__ == "__main__":
    comparator = WhisperComparison(sampling_rate=16000)
    emotion_comparator = EmotionAnalyzer()

    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    csv_filename = "../data/にじさんじ　切り抜き_20250102_202807.csv"
    search_process_df = list_original_urls(csv_filename, delete_multiple=True)
    print(search_process_df.columns)

    # チャットデータ取得
    search_process_df.to_csv(f"../urls_{current_time}.csv", index=False, encoding='utf-8-sig')
    source_urls = search_process_df["Original URL"] if "Original URL" in search_process_df.columns else search_process_df["Original videoURL"]
    clipping_urls = search_process_df["Video URL"]
    file_paths = search_process_df["File Path"]

    print(f"元配信URL: {source_urls[0]}")
    print(f"切り抜きURL: {clipping_urls[0]}")

    results = []
    progress_file = "../processing_progress.csv"
    if not os.path.exists(progress_file):
        pd.DataFrame(columns=["index", "source_url", "clipping_url", "file_path", "status"]).to_csv(
            progress_file, index=False, encoding="utf-8-sig"
        )

    for i, (source, clip, file_path) in enumerate(zip(source_urls, clipping_urls, file_paths), start=1):
        print(f"\n=== {i}/{len(source_urls)} 番目の動画の処理を開始します ===")
        matches_filename = f"../data/matches/{get_video_id_from_url(remove_query_params(source))}_{get_video_id_from_url(remove_query_params(clip))}.csv"
        if os.path.exists(matches_filename):
            print(f"File already exists: {matches_filename}. Skipping.")
            continue

        #
        source_path = download_yt_sound(source, output_dir="../data/audio/source")
        clip_path = download_yt_sound(clip, output_dir="../data/audio/clipping")
        clip_path = extract_vocals(clip_path)

        results.append({"index": i, "source_url": source, "clipping_url": clip, "file_path": file_path, "status": "文字起こし中"})
        pd.DataFrame(results).to_csv(progress_file, index=False, encoding="utf-8-sig")

        try:
            # 1. 元音声をセグメント化＆文字起こし
            source_segments = comparator.transcribe_with_vad(
                audio_file=source_path,
                threshold=0.35,
                language="ja",
                beam_size=5
            )

            # 2. 切り抜き音声をセグメント化＆文字起こし
            clipping_segments = comparator.transcribe_with_vad(
                audio_file=clip_path,
                threshold=0.35,
                language="ja",
                beam_size=5
            )

            # 3. 両者のテキストセグメントを比較
            matches, unmatched = comparator.compare_segments(
                clipping_segments,
                source_segments,
                initial_threshold=0.8,
                fast_method="sequence",
                slow_method="tfidf"
            )

            results[-1]["status"] = "成功"
            pd.DataFrame(results).to_csv(progress_file, index=False, encoding="utf-8-sig")

            # 感情分析
            analysis_result = emotion_comparator.analysis_emotion(
                file_path=file_path,
                analysis_methods=["sentiment", "weime", "mlask"],
            )
            print(f"感情分析が完了しました: {analysis_result}")

        except Exception as e:
            results[-1]["status"] = f"失敗: {str(e)}"
            pd.DataFrame(results).to_csv(progress_file, index=False, encoding="utf-8-sig")
            print(f"エラーが発生しました: {str(e)}")

    print("\n全ての動画処理が完了しました。")
    results_df = pd.DataFrame(results)
    results_df.to_csv(f"processing_{datetime.now().strftime('%Y%m%d_%H%M%S')}_results.csv", index=False, encoding="utf-8-sig")
    print("処理結果を保存しました。")
