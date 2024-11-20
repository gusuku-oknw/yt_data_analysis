import os
import traceback

import pandas as pd
import soundfile as sf
from transformers import pipeline
from datasets import load_dataset
import numpy as np
from openunmix import predict
from yt_dlp import YoutubeDL
from faster_whisper import WhisperModel
from difflib import SequenceMatcher
import subprocess
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
import librosa
from chat_download import get_video_id_from_url, remove_query_params
from pydub import AudioSegment
from charset_normalizer import detect

from audio_transcriber import AudioTranscriber


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
def compare_texts(text1, text2):
    """
    テキストの類似度を計算。

    Parameters:
        text1 (str): テキスト1。
        text2 (str): テキスト2。

    Returns:
        float: 類似度（0～1）。
    """
    return SequenceMatcher(None, text1, text2).ratio()

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
    silences = []
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

    # # ステップ3: テキストの比較
    # similarity = compare_texts(source_silences, clipping_silences)
    # print(f"セリフの類似度: {similarity * 100:.2f}%")
