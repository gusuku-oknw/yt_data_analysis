import os
import time
import librosa
import torch
from transformers import pipeline
import soundfile as sf
import numpy as np
import traceback


class AudioTranscriber:
    def __init__(self):
        self.model_id = "kotoba-tech/kotoba-whisper-v1.0"

    # 無音部分の検出
    @staticmethod
    def detect_silence(data, samplerate, threshold=0.05, min_silence_duration=0.5):
        amp = np.abs(data)  # 信号の絶対値を取得
        silence_flags = amp <= threshold  # 閾値以下が「無音」

        if silence_flags.ndim != 1:
            raise ValueError("silence_flags must be a 1D array. Check the input data.")

        silences = []  # 無音区間を記録するリスト
        prev = False  # 前のサンプルが無音かどうか
        entered = None  # 無音区間の開始位置

        for i in range(len(silence_flags)):  # silence_flags を1要素ずつ処理
            v = silence_flags[i]  # 単一の値を取得（NumPy配列内のブール値）

            if not prev and v:  # 無音が開始
                entered = i
            elif prev and not v and entered is not None:  # 無音が終了
                duration = (i - entered) / samplerate
                if duration >= min_silence_duration:  # 無音区間の長さが閾値以上の場合
                    silences.append({"from": entered, "to": i, "suffix": "cut"})
                entered = None  # 開始位置をリセット

            prev = v  # 状態を更新

        # 最後の無音区間がリストに含まれていない場合の処理
        if prev and entered is not None:
            duration = (len(silence_flags) - entered) / samplerate
            if duration >= min_silence_duration:
                silences.append({"from": entered, "to": len(silence_flags), "suffix": "cut"})

        return silences

    @staticmethod
    def get_keep_blocks(silences, data_len, samplerate, padding_time=0.2):
        keep_blocks = []

        for i, block in enumerate(silences):
            if i == 0 and block["from"] > 0:
                keep_blocks.append({"from": 0, "to": block["from"], "suffix": "keep"})
            if i > 0:
                prev = silences[i - 1]
                keep_blocks.append({"from": prev["to"], "to": block["from"], "suffix": "keep"})
            if i == len(silences) - 1 and block["to"] < data_len:
                keep_blocks.append({"from": block["to"], "to": data_len, "suffix": "keep"})

        for block in keep_blocks:
            block["from"] = max(block["from"] / samplerate - padding_time, 0)
            block["to"] = min(block["to"] / samplerate + padding_time, data_len / samplerate)

        return keep_blocks

    @staticmethod
    def save_audio_segments(data, keep_blocks, samplerate):
        output_dir = os.path.join("./content", "audio_segments_{}".format(int(time.time())))

        os.makedirs(output_dir, exist_ok=True)

        segment_files = []
        for i, block in enumerate(keep_blocks):
            start_sample = int(block["from"] * samplerate)
            end_sample = int(block["to"] * samplerate)

            if start_sample < end_sample <= len(data):
                segment_data = data[start_sample:end_sample]
                output_file = os.path.join(output_dir, f"segment_{i:02d}.wav")

                sf.write(output_file, segment_data, samplerate)
                segment_files.append(os.path.abspath(output_file))
                print(f"Saved segment {i}: {output_file}")
            else:
                print(f"Skipped invalid segment {i}: start={start_sample}, end={end_sample}")

        return segment_files

    @staticmethod
    def kotoba_whisper(audio_file):
        """
        kotoba-Whisperを使用して音声を文字起こし。

        Parameters:
            audio_file (str): 入力音声ファイルのパス。

        Returns:
            str: 文字起こし結果。
        """
        model_id = "kotoba-tech/kotoba-whisper-v1.0"
        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        model_kwargs = {"attn_implementation": "sdpa"} if torch.cuda.is_available() else {}
        generate_kwargs = {
            "task": "transcribe",
            "return_timestamps": True,
            "language": "japanese"
        }

        pipe = pipeline(
            "automatic-speech-recognition",
            model=model_id,
            torch_dtype=torch_dtype,
            device=device,
            model_kwargs=model_kwargs
        )

        audio, sampling_rate = librosa.load(audio_file, sr=16000)
        audio_input = {"raw": audio, "sampling_rate": sampling_rate}
        result = pipe(audio_input, generate_kwargs=generate_kwargs)
        print("文字起こし結果:", result["text"])
        return result["text"]

    def transcribe_audio_and_split_video(self, segment_files, keep_blocks):
        transcriptions = []

        num_segments = len(segment_files)

        for i in range(num_segments):
            block = keep_blocks[i]
            segment_start = block["from"]
            segment_end = block["to"]

            audio_file = segment_files[i]

            if not os.path.exists(audio_file):
                print(f"Audio segment file not found for block {i}: {audio_file}")
                continue

            try:
                print(f"Processing {audio_file}")
                transcription = self.kotoba_whisper(audio_file)
                transcriptions.append({
                    "text": transcription,
                    "start": segment_start,
                    "end": segment_end,
                    "audio_path": audio_file,
                })

            except Exception as e:
                error_message = f"Error during transcription for block {i}: {e}"
                print(error_message)
                print(traceback.format_exc())

                transcriptions.append({
                    "text": "",
                    "start": None,
                    "end": None,
                    "audio_path": audio_file,
                })

        return transcriptions

    def transcribe_segment(self, audio_path):
        # オーディオデータを読み込む
        data, samplerate = sf.read(audio_path)

        # ステレオデータの場合はモノラルに変換
        if data.ndim > 1:  # 2次元の場合（ステレオ音声）
            print("Stereo audio detected, converting to mono.")
            data = np.mean(data, axis=1)  # 左右チャンネルを平均化してモノラルに変換

        # 無音区間を検出
        silences = self.detect_silence(data, samplerate)
        print(f'silences: {silences}')

        # 無音区間の外側を保持するブロックを計算
        keep_blocks = self.get_keep_blocks(silences, len(data), samplerate)
        print(f'keep_blocks: {keep_blocks}')

        # 音声セグメントを保存
        segment_files = self.save_audio_segments(data, keep_blocks, samplerate)
        print(f'segment_files: {segment_files}')

        # 各セグメントを文字起こし
        transcriptions = self.transcribe_audio_and_split_video(segment_files, keep_blocks)
        print(f'transcriptions: {transcriptions}')

        return transcriptions


if __name__ == "__main__":
    try:
        transcriber = AudioTranscriber()
        audio_path = './data/sound/clipping_audio_wav/7-1fNxXj_xM.wav'
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        # 無音区間を検出
        silences = transcriber.transcribe_segment(audio_path)
        print("Detected silences:")
        for silence in silences:
            print(silence)
    except Exception as e:
        print(f"An error occurred: {e}")
        print(traceback.format_exc())
