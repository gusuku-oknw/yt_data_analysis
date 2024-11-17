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
        self.srt_list = []

    # 無音部分の検出
    @staticmethod
    def detect_silence(data, samplerate, threshold=None, min_silence_duration=0.5):
        # 閾値をRMSに基づいて設定
        if threshold is None:
            rms = np.sqrt(np.mean(data ** 2))
            threshold = rms * 0.5  # RMSの50%を閾値に設定

        amp = np.abs(data)
        silence_flags = amp > threshold
        silences = []

        prev = 1
        entered = 0
        for i, v in enumerate(silence_flags):
            if prev == 1 and v == 0:  # silence starts
                entered = i
            if prev == 0 and v == 1:  # silence ends
                duration = (i - entered) / samplerate
                if duration > min_silence_duration:
                    silences.append({"from": entered, "to": i, "suffix": "cut"})
                entered = 0
            prev = v
        return silences

    # 残すべき部分の決定
    @staticmethod
    def get_keep_blocks(silences, data_len, samplerate, padding_time=0.2):
        keep_blocks = []

        # 残す部分を決定（無音部分の前後をpadding_time秒追加）
        for i, block in enumerate(silences):
            if i == 0 and block["from"] > 0:
                keep_blocks.append({"from": 0, "to": block["from"], "suffix": "keep"})
            if i > 0:
                prev = silences[i - 1]
                keep_blocks.append({"from": prev["to"], "to": block["from"], "suffix": "keep"})
            if i == len(silences) - 1 and block["to"] < data_len:
                keep_blocks.append({"from": block["to"], "to": data_len, "suffix": "keep"})

        # 各ブロックに前後の余白を追加
        for block in keep_blocks:
            block["from"] = max(block["from"] / samplerate - padding_time, 0)
            block["to"] = min(block["to"] / samplerate + padding_time, data_len / samplerate)

        return keep_blocks

    # 音声セグメントをファイルに保存する
    @staticmethod
    def save_audio_segments(data, keep_blocks, samplerate):
        output_dir = os.path.join("./content", "audio_segments_{}".format(int(time.time())))

        # ディレクトリが存在しない場合は作成
        os.makedirs(output_dir, exist_ok=True)

        segment_files = []
        for i, block in enumerate(keep_blocks):
            start_sample = int(block["from"] * samplerate)
            end_sample = int(block["to"] * samplerate)

            # セグメントデータの確認
            if start_sample < end_sample <= len(data):
                segment_data = data[start_sample:end_sample]
                output_file = os.path.join(output_dir, f"segment_{i:02d}.wav")

                # 音声セグメントをファイルに保存
                sf.write(output_file, segment_data, samplerate)
                absolute_path = os.path.abspath(output_file)  # 絶対パスを取得
                segment_files.append(absolute_path)
                print(f"Saved segment {i}: {absolute_path}")
            else:
                print(f"Skipped invalid segment {i}: start={start_sample}, end={end_sample}")

        return segment_files

    # 文字起こしの処理
    @staticmethod
    def add_padding_and_save_segment(segment_start, segment_end, padding_time, video_file, out_dir, segment_index,
                                     file_type="video"):
        # 前後に余白を追加
        padded_start = max(segment_start - padding_time, 0)
        padded_end = segment_end + padding_time
        padded_duration = padded_end - padded_start

        # 出力ファイルパスを作成
        if file_type == "video":
            out_path = os.path.join(out_dir, f"{segment_index:02d}_padded_segment.mov")
        elif file_type == "audio":
            out_path = os.path.join(out_dir, f"{segment_index:02d}_padded_segment.wav")
        else:
            raise ValueError("Unsupported file type. Only 'video' and 'audio' are supported.")

        # FFmpegコマンドでセグメントを切り出して保存
        ffmpeg_command = [
            "ffmpeg",
            "-ss", str(padded_start),  # 開始時間（余白を加えた後）
            "-i", video_file,  # 入力動画ファイル
            "-t", str(padded_duration),  # 長さ（終了時間 - 開始時間）
            "-c", "copy",  # コーデックをコピー
            out_path  # 出力ファイルパス
        ]

        try:
            # FFmpeg コマンドを実行
            subprocess.run(ffmpeg_command, check=True)
        except subprocess.CalledProcessError as ffmpeg_error:
            print(f"FFmpeg command failed: {ffmpeg_error}")
            return None

        return out_path

    def transcribe_audio(self, segment_files, audio_file, keep_blocks, language="ja", beam_size=15,
                         temperature=0.7, save_video=False):
        """
        音声セグメントファイルを文字起こしし、対応する時間情報を含む結果を返します。

        Parameters:
            segment_files (list): 音声セグメントファイルのリスト。
            audio_file (str): 元の音声ファイル。
            keep_blocks (list): セグメントの時間情報リスト。
            language (str): 文字起こし言語。
            beam_size (int): ビームサーチのサイズ。
            temperature (float): Whisperの温度パラメータ。
            save_video (bool): 処理中の動画を保存するかどうか。

        Returns:
            list: セグメントごとの文字起こし結果リスト。
        """
        transcriptions = []

        # セグメントとブロックの最小長を使用
        num_segments = min(len(segment_files), len(keep_blocks))

        for i in range(num_segments):
            block = keep_blocks[i]
            segment_file = segment_files[i]

            # セグメントの情報を取得
            segment_start = block.get("from", 0.0)
            segment_end = block.get("to", 0.0)

            # 音声ファイルの存在を確認
            if not os.path.exists(segment_file):
                print(f"[WARNING] Segment file not found for block {i}: {segment_file}")
                continue

            try:
                # 音声セグメントを文字起こし
                transcription = self.process_segment(segment_file, segment_start, segment_end)
                transcriptions.append(transcription)

            except Exception as e:
                # エラー時の処理を追加
                print(f"[ERROR] Error during transcription for block {i}: {e}")
                transcriptions.append({
                    "file_path": os.path.abspath(segment_file) if os.path.exists(segment_file) else None,
                    "text": "",
                    "start": None,
                    "duration": None,
                    "audio": {
                        "audio_path": segment_file if os.path.exists(segment_file) else None,
                        "start_time": segment_start,
                        "end_time": segment_end,
                    }
                })

        return transcriptions

    def process_segment(self, segment_file, segment_start, segment_end, padding_time=0.001):
        """音声セグメントを処理して文字起こし結果を生成します。"""
        print(f"[INFO] Processing file: {segment_file}")
        segments = self.pipe(segment_file, generate_kwargs=self.generate_kwargs)
        text = segments['text']

        return {
            "text": text,
            "start": segment_start,
            "end": segment_end,
            "file_path": segment_file
        }

    def transcribe_segment(self, audio_path, language="ja", beam_size=15, temperature=0.7):
        # 1. 音声ファイルのパスを取得
        data, samplerate = sf.read(audio_path)
        # audio = AudioSegment.from_wav(audio_path)
        # audio = audio.set_channels(1)
        # print(audio)

        # 2. 無音部分を検出
        silences = self.detect_silence(data, samplerate)
        print(f'silences: {silences}')

        # 3. 残すべき部分を決定
        keep_blocks = self.get_keep_blocks(silences, len(data), samplerate)
        print(f'keep_blocks: {keep_blocks}')

        # 4. 残すべき音声セグメントをファイルに保存
        segment_files = self.save_audio_segments(data, keep_blocks, samplerate)
        print(f'segment_files: {segment_files}')

        # 5. 文字起こしの実行
        transcriptions = self.transcribe_audio(segment_files, audio_path, keep_blocks,
                                                               language=language,
                                                               beam_size=beam_size,
                                                               temperature=temperature)
        print(f'transcriptions: {transcriptions}')

        return transcriptions
