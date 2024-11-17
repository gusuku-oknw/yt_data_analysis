import soundfile as sf
import subprocess
import numpy as np
import os
import time
from dotenv import load_dotenv
from tqdm import tqdm
import torch
from transformers import pipeline
import traceback


def seconds_to_hhmmssms(seconds):
    milliseconds = int(seconds * 1000)
    hours, milliseconds = divmod(milliseconds, 3600000)
    minutes, milliseconds = divmod(milliseconds, 60000)
    seconds, milliseconds = divmod(milliseconds, 1000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"


class AudioTranscriber:
    def __init__(self):
        # モデルの設定
        model_id = "kotoba-tech/kotoba-whisper-v1.0"
        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        model_kwargs = {"attn_implementation": "sdpa"} if torch.cuda.is_available() else {}
        self.generate_kwargs = {
            "task": "transcribe",
            "return_timestamps": True,
            "language": "japanese"  # 日本語指定
        }

        # モデルのロード
        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=model_id,
            torch_dtype=torch_dtype,
            device=device,
            model_kwargs=model_kwargs
        )
        self.output_dir = "./content"
        self.srt_list = []

    @staticmethod
    def detect_silence(data, samplerate, threshold=None, min_silence_duration=0.5):
        """
        無音部分を検出する関数
        """
        if threshold is None:
            rms = np.sqrt(np.mean(data ** 2))
            threshold = rms * 0.5  # RMSの50%を閾値に設定

        amp = np.abs(data)
        silence_flags = amp < threshold
        silence_indices = np.where(silence_flags)[0]
        silence_durations = np.diff(np.insert(silence_indices, [0, len(silence_indices)], [0, len(data)]))

        # 無音区間を計算
        silence_blocks = []
        start_idx = None
        for idx, duration in enumerate(silence_durations):
            if duration > samplerate * min_silence_duration:
                if start_idx is None:
                    start_idx = silence_indices[idx]
                else:
                    silence_blocks.append({"from": start_idx, "to": silence_indices[idx - 1]})
                    start_idx = None
        return silence_blocks

    @staticmethod
    def get_keep_blocks(silences, data_len, samplerate, padding_time=0.2):
        """
        無音区間を除外した音声ブロックを取得
        """
        keep_audio_blocks = []
        for i, block in enumerate(silences):
            if i == 0 and block["from"] > 0:
                keep_audio_blocks.append({"from": 0, "to": block["from"]})
            if i > 0:
                prev = silences[i - 1]
                keep_audio_blocks.append({"from": prev["to"], "to": block["from"]})
            if i == len(silences) - 1 and block["to"] < data_len:
                keep_audio_blocks.append({"from": block["to"], "to": data_len})

        # 無音区間の前後にパディングを追加
        for block in keep_audio_blocks:
            block["from"] = max(block["from"] / samplerate - padding_time, 0)
            block["to"] = min(block["to"] / samplerate + padding_time, data_len / samplerate)
        return keep_audio_blocks

    @staticmethod
    def save_audio_segments(data, keep_audio_blocks, samplerate):
        """
        音声ブロックを個別ファイルに保存
        """
        timestamp = int(time.time())
        output_dir = os.path.join("./content", f"audio_segments_{timestamp}")
        os.makedirs(output_dir, exist_ok=True)

        segment_files = []
        for i, block in enumerate(keep_audio_blocks):
            start_sample = int(block["from"] * samplerate)
            end_sample = int(block["to"] * samplerate)
            if 0 <= start_sample < len(data) and start_sample < end_sample:
                segment_data = data[start_sample:end_sample]
                output_file = os.path.join(output_dir, f"segment_{i:02d}.wav")
                sf.write(output_file, segment_data, samplerate)
                segment_files.append(output_file)
        return segment_files

    def transcribe_audio(self, segment_files, keep_audio_blocks):
        """
        音声ファイルを処理して文字起こし
        """
        transcriptions = []
        for i, (segment_file, block) in enumerate(tqdm(zip(segment_files, keep_audio_blocks), total=len(segment_files))):
            try:
                transcription = self.process_segment(segment_file, block["from"], block["to"])
                transcriptions.append(transcription)
            except Exception as e:
                print(f"[ERROR] Transcription failed for segment {i}: {e}")
        return transcriptions

    def process_segment(self, segment_file, segment_start, segment_end):
        """
        音声セグメントを処理してテキスト化
        """
        result = self.pipe(segment_file, generate_kwargs=self.generate_kwargs)
        return {
            "text": result['text'],
            "start": segment_start,
            "end": segment_end,
            "file_path": segment_file
        }

    def transcribe_segment(self, audio_path):
        """
        音声ファイル全体の文字起こしを実行
        """
        data, samplerate = sf.read(audio_path)
        silences = self.detect_silence(data, samplerate)
        keep_audio_blocks = self.get_keep_blocks(silences, len(data), samplerate)
        segment_files = self.save_audio_segments(data, keep_audio_blocks, samplerate)
        return self.transcribe_audio(segment_files, keep_audio_blocks)


if __name__ == "__main__":
    try:
        transcriber = AudioTranscriber()
        audio_path = './content/2022-03-08_14-19-00.wav'
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        transcriptions = transcriber.transcribe_segment(audio_path)
        for transcription in transcriptions:
            print(transcription)
    except Exception as e:
        print(f"An error occurred: {e}")
        print(traceback.format_exc())
