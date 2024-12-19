import os
from asyncio import as_completed
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import numpy as np
import soundfile as sf
from transformers import pipeline
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter
from matplotlib import rcParams
from yt_dlp import YoutubeDL
from faster_whisper import WhisperModel
import torch
import librosa
from chat_download import get_video_id_from_url, remove_query_params
from pydub import AudioSegment
from charset_normalizer import detect
from audio_transcriber import AudioTranscriber
from janome.tokenizer import Tokenizer
from difflib import SequenceMatcher
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# Janomeトークナイザーの初期化
janome_tokenizer = Tokenizer()

def preprocess_text(text):
    """テキストの前処理（スペース削除、記号削除など）"""
    import re
    text = re.sub(r'\s+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    return text

def tokenize_japanese(text):
    tokens = [token.surface for token in janome_tokenizer.tokenize(text)]
    return tokens

def calculate_similarity(text1, text2, method="sequence"):
    text1 = preprocess_text(text1)
    text2 = preprocess_text(text2)

    if not text1.strip() or not text2.strip():
        return 0.0

    if method == "sequence":
        return SequenceMatcher(None, text1, text2).ratio()
    elif method == "jaccard":
        set1, set2 = set(tokenize_japanese(text1)), set(tokenize_japanese(text2))
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        return intersection / union if union != 0 else 0
    elif method == "tfidf":
        vectorizer = TfidfVectorizer(tokenizer=tokenize_japanese)
        try:
            tfidf_matrix = vectorizer.fit_transform([text1, text2]).toarray()
            numerator = np.dot(tfidf_matrix[0], tfidf_matrix[1])
            denominator = np.linalg.norm(tfidf_matrix[0]) * np.linalg.norm(tfidf_matrix[1])
            return numerator / denominator if denominator != 0 else 0.0
        except ValueError:
            return 0.0
    else:
        raise ValueError("Unsupported similarity method")


def find_best_match(segment, source_segments, used_segments, threshold=0.8, method="tfidf"):
    """切り抜きセグメントに最も類似する元動画セグメントを探す"""
    best_match = None
    max_similarity = 0

    for src in source_segments:
        # 使用済みセグメントはスキップ
        if src["start"] in used_segments:
            continue

        similarity = calculate_similarity(segment["text"], src["text"], method=method)
        if similarity > max_similarity and similarity >= threshold:
            best_match = src
            max_similarity = similarity

    return best_match

def compare_segments(clipping_segments, source_segments, initial_threshold=0.8, fast_method="sequence", slow_method="tfidf"):
    """
    切り抜きセグメント(clipping_segments)と元動画セグメント(source_segments)を比較し、最適なマッチングを行います。
    まずfast_methodでマッチングを試み、マッチしなかったものについてのみslow_methodで再マッチングを行います。
    並列処理を用いてパフォーマンスを向上させます。
    """

    matches = []
    unmatched = []
    used_segments = set()

    def process_clip(clip, method):
        try:
            threshold = initial_threshold
            found_match = False
            clip_matches = []

            # 長いテキストを50文字ごとに分割
            long_segments = [clip["text"]]
            if len(clip["text"]) > 50:
                long_segments = [clip["text"][i:i + 50] for i in range(0, len(clip["text"]), 50)]
            # 空白だけのセグメントは削除
            long_segments = [seg for seg in long_segments if seg.strip()]

            for segment_text in long_segments:
                local_threshold = threshold
                while local_threshold > 0.1:
                    best_match = find_best_match({"text": segment_text}, source_segments, used_segments, local_threshold, method=method)
                    if best_match:
                        clip_matches.append({
                            "clip_text": segment_text,
                            "clip_start": clip["start"],
                            "clip_end": clip["end"],
                            "source_text": best_match["text"],
                            "source_start": best_match["start"],
                            "source_end": best_match["end"],
                            "similarity": calculate_similarity(segment_text, best_match["text"], method=method),
                        })
                        used_segments.add(best_match["start"])
                        found_match = True
                        break
                    local_threshold -= 0.05

            if not found_match:
                # このクリップはマッチングできなかった
                return {
                    "clip_text": clip["text"],
                    "clip_start": clip["start"],
                    "clip_end": clip["end"],
                    "matched": False
                }

            return {"matches": clip_matches, "matched": True}

        except Exception as e:
            print(f"Error processing clip '{clip.get('text', 'Unknown')}': {e}")
            return {"error": str(e), "matched": False}

    # Step 1: Fast methodでのマッチング
    print("Fast methodでのマッチングを実行中...")
    with ThreadPoolExecutor() as executor:
        futures = {executor.submit(process_clip, clip, fast_method): clip for clip in clipping_segments}
        for future in tqdm(as_completed(futures), total=len(clipping_segments), desc="Comparing segments with fast_method"):
            result = future.result()
            if result.get("matched"):
                matches.extend(result["matches"])
            elif "clip_text" in result:
                unmatched.append(result)

    # Step 2: Slow methodでの再マッチング（unmatchedに対してのみ）
    if unmatched:
        print("Unmatchedに対してslow methodでのマッチングを実行中...")
        still_unmatched = []
        with ThreadPoolExecutor() as executor:
            futures = {executor.submit(process_clip, u, slow_method): u for u in unmatched}
            for future in tqdm(as_completed(futures), total=len(unmatched), desc="Comparing segments with slow_method"):
                result = future.result()
                if result.get("matched"):
                    matches.extend(result["matches"])
                elif "clip_text" in result:
                    still_unmatched.append(result)

        unmatched = still_unmatched

    return matches, unmatched

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
    video_id = get_video_id_from_url(remove_query_params(url))
    file_name = video_id  # 拡張子はFFmpegで付加
    file_path_no_ext = os.path.join(output_dir, file_name)
    file_path_with_ext = f"{file_path_no_ext}.mp3"

    # ファイルが存在する場合はダウンロードをスキップ
    if os.path.exists(file_path_with_ext):
        print(f"ファイルは既に存在します: {file_path_with_ext}")
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

# 音声ファイルをWAV形式に変換
def convert_to_wav(audio_file, output_dir):
    """
    音声ファイルをWAV形式に変換（pydubを使用）。

    Parameters:
        audio_file (str): 入力音声ファイル。
        output_dir (str): 出力ディレクトリ。

    Returns:
        str: 変換後のWAVファイルのパス。
    """
    # 出力ディレクトリを作成（存在しない場合）
    os.makedirs(output_dir, exist_ok=True)

    # 出力ファイルパスを生成
    wav_file = os.path.join(output_dir, os.path.splitext(os.path.basename(audio_file))[0] + ".wav")

    # 音声ファイルを読み込み、WAV形式で保存
    audio = AudioSegment.from_file(audio_file)
    audio.export(wav_file, format="wav")

    return wav_file

# ボーカル音声のパスを取得（UMX）
def get_demucs_output_path(output_dir, audio_file):
    """
    Demucsの分離後のボーカルファイルのパスを取得。

    Parameters:
        output_dir (str): Demucsの出力ディレクトリ。
        audio_file (str): 入力音声ファイルのパス。

    Returns:
        str: ボーカル音声（vocals.wav）のパス。
    """
    # 入力ファイル名（拡張子なし）を取得
    base_name = os.path.splitext(os.path.basename(audio_file))[0]
    vocals_path = os.path.join(output_dir, "htdemucs", base_name, "vocals.wav")

    if not os.path.exists(vocals_path):
        raise FileNotFoundError(f"指定されたファイルが見つかりません: {vocals_path}")

    return vocals_path

def audio_transcription2csv(audio_path, output_directory, extract=True):
    """
    音声ファイルを文字起こしし、結果をCSVファイルに保存または既存のCSVを読み込み。

    Parameters:
        audio_path (str): 入力音声ファイルのパス。
        output_directory (str): 出力CSVファイルのディレクトリ。

    Returns:
        list: 文字起こし結果（辞書形式のリスト）。
    """
    # ファイル名を適切に生成
    output_path = os.path.join(
        output_directory, os.path.basename(audio_path).replace('.mp3', '.csv')
    )

    # CSVファイルが存在する場合、エンコーディングを検出して読み込む
    if os.path.exists(output_path):
        print(f"CSVファイルが既に存在します: {output_path} を読み込みます...")

        # ファイルのエンコーディングを検出
        with open(output_path, 'rb') as f:
            detected = detect(f.read())
            encoding = detected['encoding']
            print(f"検出されたエンコーディング: {encoding}")

        # 正しいエンコーディングでCSVを読み込む
        df = pd.read_csv(output_path, encoding=encoding)
        silences = df.to_dict(orient="records")
        return silences

    # CSVファイルが存在しない場合、新たに作成
    print(f"CSVファイルが存在しません: {output_path} を作成します...")

    # 音声ファイルの処理（文字起こし）
    transcriber = AudioTranscriber()
    if not os.path.exists(os.path.abspath(audio_path)):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    # 無音区間と文字起こしを行う
    silences = transcriber.transcribe_segment(os.path.abspath(audio_path), audio_extract=extract)
    print("Detected silences:", silences)

    # データフレームに変換
    df = pd.DataFrame(silences)

    # CSVファイルに保存
    os.makedirs(output_directory, exist_ok=True)  # 出力ディレクトリの作成
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"CSVファイルに保存しました: {output_path}")


    return silences

def plt_compare(matches_csv, unmatched_csv):
    # 日本語フォントの設定
    rcParams['font.family'] = 'Meiryo'  # 日本語フォントを指定（インストール済みである必要があります）

    # 一致したセグメントの読み込み
    matches_df = pd.read_csv(matches_csv)

    # 一致しなかったセグメントの読み込み
    if os.path.exists(unmatched_csv):
        unmatched_df = pd.read_csv(unmatched_csv)
    else:
        unmatched_df = pd.DataFrame()

    # 時間オフセットの計算
    matches_df['time_offset'] = matches_df['source_start'] - matches_df['clip_start']
    average_offset = matches_df['time_offset'].mean()

    # 秒を「時:分:秒」の形式に変換する関数
    def seconds_to_hms(seconds):
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours:02}:{minutes:02}:{secs:02}"

    # プロットを初期化
    plt.figure(figsize=(15, 6))

    # 2つのサブプロットを作成
    ax1 = plt.subplot(2, 1, 1)  # 切り抜き音声
    ax2 = plt.subplot(2, 1, 2)  # 元音声

    # 切り抜き音声のプロット
    for _, row in matches_df.iterrows():
        ax1.barh(y=0, width=row['clip_end'] - row['clip_start'],
                 left=row['clip_start'], height=0.4, align='center', color='green')

    if not unmatched_df.empty:
        for _, row in unmatched_df.iterrows():
            ax1.barh(y=0, width=row['clip_end'] - row['clip_start'],
                     left=row['clip_start'], height=0.4, align='center', color='red')

    ax1.set_title('切り抜き音声のセグメント', fontsize=14)
    ax1.set_xlim(0, matches_df['clip_end'].max())
    ax1.set_yticks([])

    # 時間表示を時:分:秒形式に設定
    ax1.xaxis.set_major_formatter(FuncFormatter(lambda x, _: seconds_to_hms(x)))
    ax1.set_xlabel('時間（時:分:秒）', fontsize=12)

    # 元音声のプロット
    for _, row in matches_df.iterrows():
        ax2.barh(y=0, width=row['source_end'] - row['source_start'],
                 left=row['source_start'], height=0.4, align='center', color='green')

    # if not unmatched_df.empty:
    #     for _, row in unmatched_df.iterrows():
    #         # 推定された元音声上のセグメントの位置を計算
    #         estimated_start = row['clip_start'] + average_offset
    #         estimated_end = row['clip_end'] + average_offset
    #         ax2.barh(y=0, width=estimated_end - estimated_start,
    #                  left=estimated_start, height=0.4, align='center', color='red', alpha=0.5)

    ax2.set_title('元音声のセグメント', fontsize=14)
    ax2.set_xlim(0, matches_df['source_end'].max())
    ax2.set_yticks([])

    # 時間表示を時:分:秒形式に設定
    ax2.xaxis.set_major_formatter(FuncFormatter(lambda x, _: seconds_to_hms(x)))
    ax2.set_xlabel('時間（時:分:秒）', fontsize=12)

    # 凡例の追加
    legend_elements = [
        Patch(facecolor='green', label='一致したセグメント'),
        Patch(facecolor='red', label='一致しなかったセグメント')
    ]
    ax1.legend(handles=legend_elements, loc='upper right', fontsize=10)
    ax2.legend(handles=[Patch(facecolor='green', label='一致したセグメント')], loc='upper right', fontsize=10)

    # レイアウトを調整して表示
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # # 元配信URLと切り抜きURL
    # source_url = "https://www.youtube.com/live/YGGLxywB3Tw?si=O5Aa-5KqFPqQD8Xd"
    # clipping_url = "https://www.youtube.com/watch?v=7-1fNxXj_xM"
    #
    # # ステップ1: 音声ダウンロード
    # print("元配信音声をダウンロード中...")
    # source_audio = download_yt_sound(source_url, output_dir="data/sound/source_audio")
    # # print(source_audio)
    # print("切り抜き音声をダウンロード中...")
    # clipping_audio = download_yt_sound(clipping_url, output_dir="data/sound/clipping_audio")
    # # print(clipping_audio)
    # test = "./data"

    source_audio = "../data/sound/source_audio/pnHdRQbR2zs.mp3"
    clipping_audio = "../data/sound/clipping_audio/-bRcKCM5_3E.mp3"

    # ステップ2: Distil-Whisperで文字起こし
    print("元配信音声を文字起こし中...")
    source_silences = audio_transcription2csv(
        source_audio,
        output_directory="../data/transcription/source",
        extract=False
    )
    print(f"元配信文字起こし結果: {source_silences}")

    # 切り抜き音声を文字起こししてCSVに保存
    print("切り抜き音声を文字起こし中...")
    clipping_silences = audio_transcription2csv(
        clipping_audio,
        output_directory="../data/transcription/clipping"
    )
    print(f"切り抜き文字起こし結果: {clipping_silences}")

    # ステップ3: テキストの比較
    root_directory = '../data/compare_CSV'
    print(len(source_silences), len(clipping_silences))
    matches, unmatched = compare_segments(clipping_silences, source_silences)

    file_name_basename = \
        (f"{os.path.basename(source_audio).replace('.wav', '').replace('.mp3', '')}"
         f"_"
         f"{os.path.basename(clipping_audio).replace('.wav', '').replace('.mp3', '')}"
         f".csv")

    # ファイル名を作成
    output_file = os.path.join(root_directory,
                               f"{file_name_basename}")
    unmatched_file = output_file.replace(".csv", "_unmatched.csv")

    # CSVに保存
    os.makedirs(root_directory, exist_ok=True)

    pd.DataFrame(matches).to_csv(output_file, index=False, encoding="utf-8-sig")
    print(f"結果をCSVファイルに保存しました: {output_file}")

    if unmatched:
        pd.DataFrame(unmatched).to_csv(unmatched_file, index=False, encoding="utf-8-sig")
        print(f"一致しなかったセグメントを保存しました: {unmatched_file}")

    plt_compare(output_file, unmatched_file)
