import os
from yt_dlp import YoutubeDL
from faster_whisper import WhisperModel
from difflib import SequenceMatcher
import subprocess
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
import librosa


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

    ydl_opts = {
        'format': 'bestaudio' if audio_only else 'bestvideo+bestaudio',
        'outtmpl': os.path.join(output_dir, '%(title)s.%(ext)s'),
        'noplaylist': True,
        'quiet': False,
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }] if audio_only else None
    }

    with YoutubeDL(ydl_opts) as ydl:
        info_dict = ydl.extract_info(url, download=True)
        return ydl.prepare_filename(info_dict)


def separate_audio_with_demucs(audio_file, output_dir):
    """
    Demucsを使用して音声を分離。

    Parameters:
        audio_file (str): 入力音声ファイル。
        output_dir (str): 分離後の音声保存先ディレクトリ。
    """
    os.makedirs(output_dir, exist_ok=True)

    try:
        command = ['demucs', '-o', output_dir, audio_file]
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Demucs分離エラー: {e}")
        raise


def distil_whisper(audio_file):
    """
    Distil-Whisperを使用して音声を文字起こし。

    Parameters:
        audio_file (str): 入力音声ファイルのパス。

    Returns:
        str: 文字起こし結果。
    """
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model_id = "distil-whisper/distil-large-v3"

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True
    ).to(device)

    processor = AutoProcessor.from_pretrained(model_id)
    audio, _ = librosa.load(audio_file, sr=16000)

    inputs = processor(audio, return_tensors="pt", sampling_rate=16000).to(device)
    with torch.no_grad():
        generated_ids = model.generate(inputs["input_features"])
    return processor.batch_decode(generated_ids, skip_special_tokens=True)[0]


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


if __name__ == "__main__":
    # 元配信URLと切り抜きURL
    source_url = "https://www.youtube.com/watch?v=SOURCE_VIDEO_ID"
    clipping_url = "https://www.youtube.com/watch?v=CLIPPING_VIDEO_ID"

    # ステップ1: 音声ダウンロード
    source_audio = download_yt_sound(source_url, audio_only=True)
    clipping_audio = download_yt_sound(clipping_url, audio_only=True)

    # ステップ2: Demucsで音声分離
    source_output_dir = "output/source_audio"
    clipping_output_dir = "output/clipping_audio"
    separate_audio_with_demucs(source_audio, source_output_dir)
    separate_audio_with_demucs(clipping_audio, clipping_output_dir)

    # 分離後のセリフ音声
    source_vocals = os.path.join(source_output_dir, "htdemucs", "vocals.wav")
    clipping_vocals = os.path.join(clipping_output_dir, "htdemucs", "vocals.wav")

    # ステップ3: Distil-Whisperで文字起こし
    source_text = distil_whisper(source_vocals)
    clipping_text = distil_whisper(clipping_vocals)

    # ステップ4: テキストの比較
    similarity = compare_texts(source_text, clipping_text)
    print(f"セリフの類似度: {similarity * 100:.2f}%")
