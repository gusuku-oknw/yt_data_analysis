import os
import traceback

import pandas as pd
import soundfile as sf
from transformers import pipeline
from datasets import load_dataset
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter
from matplotlib import rcParams

import numpy as np
from openunmix import predict
from yt_dlp import YoutubeDL
from faster_whisper import WhisperModel
import subprocess
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
import librosa
from chat_download import get_video_id_from_url, remove_query_params
from pydub import AudioSegment
from charset_normalizer import detect
from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from difflib import SequenceMatcher

from audio_transcriber import AudioTranscriber
from sample_codes.spleeter_test import root_directory


# 音声ファイルを文字起こし
def kotoba_whisper(audio_file):
    """
    kotoba-Whisperを使用して音声を文字起こし。

    Parameters:
        audio_file (str): 入力音声ファイルのパス。

    Returns:
        str: 文字起こし結果。
    """
    # モデルの設定
    model_id = "kotoba-tech/kotoba-whisper-v1.0"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model_kwargs = {"attn_implementation": "sdpa"} if torch.cuda.is_available() else {}
    generate_kwargs = {
        "task": "transcribe",
        "return_timestamps": True,
        "language": "japanese"  # 日本語指定
    }

    # モデルのロード
    pipe = pipeline(
        "automatic-speech-recognition",
        model=model_id,
        torch_dtype=torch_dtype,
        device=device,
        model_kwargs=model_kwargs
    )

    # 音声ファイルの読み込み
    audio, sampling_rate = librosa.load(audio_file, sr=16000)

    # 音声データを辞書形式に変換
    audio_input = {"raw": audio, "sampling_rate": sampling_rate}

    # 推論の実行
    result = pipe(audio_input, generate_kwargs=generate_kwargs)

    # 文字起こし結果を表示
    print("文字起こし結果:", result["text"])
    return result["text"]

# 音声ファイルを文字起こし（fast-whisper）
def fast_whisper_transcription(audio_file, model_size="base", device="cuda", compute_type="float16"):
    """
    fast-whisper を使用して音声を文字起こし。

    Parameters:
        audio_file (str): 入力音声ファイルのパス。
        model_size (str): 使用するモデルサイズ（例: "large-v2", "medium", "small"）。
        device (str): 使用するデバイス（例: "cuda", "cpu"）。
        compute_type (str): 演算の型（例: "float16", "float32"）。

    Returns:
        str: 文字起こし結果。
    """
    # モデルのロード
    model = WhisperModel(model_size, device=device, compute_type=compute_type)

    # 音声データのサンプリングレートを確認し、必要なら変換
    audio, _ = librosa.load(audio_file, sr=16000)  # サンプリングレートを16kHzに統一
    temp_wav = f"{os.path.splitext(audio_file)[0]}_temp.wav"

    # 音声データを一時ファイルに保存（soundfileを使用）
    sf.write(temp_wav, audio, samplerate=16000)

    # 文字起こしの実行
    segments, _ = model.transcribe(temp_wav, language="ja")

    # 一時ファイルを削除
    os.remove(temp_wav)

    # 文字起こし結果を連結して返す
    transcription = " ".join([segment.text for segment in segments])
    return transcription

# テキストの類似度を計算
def compare_segments(clipping_segments, source_segments, initial_threshold=1.0, time_margin=30.0):
    """
    切り抜き動画と元動画を一致させる（並列処理を使用）。
    結果をCSVファイルに保存。

    Parameters:
        clipping_segments (list): 切り抜き動画のセグメントリスト。
        source_segments (list): 元動画のセグメントリスト。
        initial_threshold (float): 初期の類似度しきい値。
        time_margin (float): 探索範囲の時間（秒）。

    Returns:
        list: 一致したセグメントペアのリスト。
    """
    def calculate_similarity(text1, text2, method="sequence"):
        """テキスト間の類似度を計算"""
        try:
            if method == "sequence":
                return SequenceMatcher(None, text1, text2).ratio()
            elif method == "jaccard":
                set1 = set(text1.split())
                set2 = set(text2.split())
                intersection = len(set1 & set2)
                union = len(set1 | set2)
                return intersection / union if union != 0 else 0
            elif method == "cosine":
                vectorizer = CountVectorizer().fit_transform([text1, text2])
                vectors = vectorizer.toarray()
                return cosine_similarity(vectors)[0, 1]
        except Exception as e:
            print(f"Error in calculate_similarity: {e}")
            return 0

    def find_best_match(segment, source_segments, threshold=0.8):
        """切り抜きセグメントに最も類似する元動画セグメントを探す"""
        best_match = None
        max_similarity = 0
        for src in source_segments:
            similarity = calculate_similarity(segment["text"], src["text"])
            if similarity > max_similarity and similarity >= threshold:
                best_match = src
                max_similarity = similarity
        return best_match

    def process_clip(clip):
        nonlocal unmatched
        try:
            threshold = initial_threshold
            while threshold > 0.5:
                # Remove time-based filtering
                filtered_segments = source_segments
                best_match = find_best_match(clip, filtered_segments, threshold)
                if best_match:
                    return {
                        "clip_text": clip["text"],
                        "clip_start": clip["start"],
                        "clip_end": clip["end"],
                        "source_text": best_match["text"],
                        "source_start": best_match["start"],
                        "source_end": best_match["end"],
                        "similarity": calculate_similarity(clip["text"], best_match["text"]),
                    }
                threshold -= 0.1
            unmatched.append({"clip_text": clip["text"], "clip_start": clip["start"], "clip_end": clip["end"]})
        except Exception as e:
            print(f"Error processing clip '{clip['text']}': {e}")

    # 並列処理の実行
    matches = []
    unmatched = []  # 一致しなかったセグメントを記録するリスト

    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_clip, clip) for clip in clipping_segments]
        for future in as_completed(futures):
            try:
                result = future.result()
                if result:
                    matches.append(result)
            except Exception as e:
                print(f"Error in thread: {e}")

    return matches, unmatched

# YouTube動画または音声をダウンロード
def download_yt_sound(url, output_dir="data/sound", audio_only=True):
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

    # 出力ファイル名を生成（拡張子を確認して付与）
    video_id = get_video_id_from_url(remove_query_params(url))
    file_name = f"{video_id}.mp3"
    file_path = os.path.join(output_dir, file_name)

    ydl_opts = {
        'format': 'bestaudio' if audio_only else 'bestvideo+bestaudio',
        'outtmpl': file_path,
        'noplaylist': True,
        'quiet': False,
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }] if audio_only else None
    }

    with YoutubeDL(ydl_opts) as ydl:
        ydl.extract_info(url, download=True)

    # ダウンロードされたファイルの確認と修正
    if not file_path.endswith(".mp3"):
        file_path = f"{file_path}.mp3"

    return file_path

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

def audio_transcription2csv(audio_path, output_directory):
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
        output_directory, os.path.basename(audio_path).replace('.wav', '_transcription.csv')
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
    silences = transcriber.transcribe_segment(os.path.abspath(audio_path))
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
    # 元配信URLと切り抜きURL
    source_url = "https://www.youtube.com/live/YGGLxywB3Tw?si=O5Aa-5KqFPqQD8Xd"
    clipping_url = "https://www.youtube.com/watch?v=7-1fNxXj_xM"

    # ステップ1: 音声ダウンロード
    print("元配信音声をダウンロード中...")
    # source_audio = download_yt_sound(source_url, output_dir="data/sound/source_audio")
    # print(source_audio)
    print("切り抜き音声をダウンロード中...")
    # clipping_audio = download_yt_sound(clipping_url, output_dir="data/sound/clipping_audio")
    # print(clipping_audio)
    test = "./data"
    # test
    source_audio = "./data/sound/source_audio_wav/O5Aa-5KqFPqQD8Xd.wav"
    clipping_audio = "./data/sound/clipping_audio_wav/7-1fNxXj_xM.wav"

    # ステップ2: Distil-Whisperで文字起こし
    print("元配信音声を文字起こし中...")
    source_silences = audio_transcription2csv(
        source_audio,
        output_directory="data/sound/source_audio_transcription"
    )
    print(f"元配信文字起こし結果: {source_silences}")

    # 切り抜き音声を文字起こししてCSVに保存
    print("切り抜き音声を文字起こし中...")
    clipping_silences = audio_transcription2csv(
        clipping_audio,
        output_directory="data/sound/clipping_audio_transcription"
    )
    print(f"切り抜き文字起こし結果: {clipping_silences}")

    # ステップ3: テキストの比較
    root_directory = 'data/compare_CSV'
    print(len(source_silences), len(clipping_silences))
    matches, unmatched = compare_segments(clipping_silences, source_silences)

    # ファイル名を作成
    output_file = os.path.join(root_directory,
                               f"{os.path.basename(source_audio).replace('.wav', '')}_{os.path.basename(clipping_audio).replace('.wav', '')}.csv")
    unmatched_file = output_file.replace(".csv", "_unmatched.csv")

    # CSVに保存
    os.makedirs(root_directory, exist_ok=True)

    pd.DataFrame(matches).to_csv(output_file, index=False, encoding="utf-8-sig")
    print(f"結果をCSVファイルに保存しました: {output_file}")

    if unmatched:
        pd.DataFrame(unmatched).to_csv(unmatched_file, index=False, encoding="utf-8-sig")
        print(f"一致しなかったセグメントを保存しました: {unmatched_file}")

    plt_compare(output_file, unmatched_file)
