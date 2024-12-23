import json
import os
import time
import librosa
import torch
from transformers import pipeline
import soundfile as sf
import numpy as np
import traceback
import subprocess
from tqdm import tqdm
from pyannote.audio import Pipeline
from moviepy.editor import *
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.environ['HUGGINGFACE_ACCESS_TOKEN_GOES_HERE']


class AudioTranscriber:
    def __init__(self):
        self.model_id = "kotoba-tech/kotoba-whisper-v1.0"

    @staticmethod
    def speaker_separation(audio_file, video_file, output_dir="../content/speaker_segments", sampling_rate=16000):
        """
        Pyannote Audio を使用してスピーカー分離を行い、各話者の音声区間を保存。

        Parameters:
            audio_file (str): 入力オーディオファイルのパス。
            video_file (str): 入力動画ファイルのパス（動画タイムラインに基づいて調整）。
            output_dir (str): 分離された音声ファイルの保存先ディレクトリ。
            sampling_rate (int): サンプリングレート (16kHzを推奨)。

        Returns:
            list: 分離された音声区間情報のリスト。
                  各要素は {"speaker": "SpeakerID", "start": float, "end": float, "audio_path": str} の形式。
        """
        try:
            # 話者分離パイプラインをロード
            pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization",
                use_auth_token="YOUR_ACCESS_TOKEN"  # 必ずトークンを取得して設定
            )

            # 話者分離の結果を取得
            diarization = pipeline(audio_file)

            # 出力ディレクトリを準備
            os.makedirs(output_dir, exist_ok=True)

            # 動画の長さを取得
            video_clip = VideoFileClip(video_file)
            video_duration = video_clip.duration

            # 区間ごとの情報をリストに格納
            speaker_segments = []

            # librosaで音声データをロード（16kHz強制変換）
            audio_data, sr = librosa.load(audio_file, sr=sampling_rate, mono=False)

            # ステレオをモノラルに変換
            if audio_data.ndim == 2:
                audio_data = audio_data.mean(axis=0)

            for i, (turn, _, speaker) in enumerate(diarization.itertracks(yield_label=True)):
                # タイムラインを動画に基づいて調整
                start_time = max(0, turn.start)
                end_time = min(video_duration, turn.end)

                print(f"start={start_time:.1f}s stop={end_time:.1f}s speaker_{speaker}")

                # 話者ごとの区間を保存
                start_sample = int(start_time * sampling_rate)
                end_sample = int(end_time * sampling_rate)
                segment_data = audio_data[start_sample:end_sample]

                # ファイルに保存
                output_file = os.path.join(output_dir, f"speaker_{speaker}_{i:02d}.wav")
                sf.write(output_file, segment_data, samplerate=sampling_rate)

                # 結果をリストに追加
                speaker_segments.append({
                    "speaker": speaker,
                    "start": start_time,
                    "end": end_time,
                    "audio_path": output_file
                })

            return speaker_segments

        except Exception as e:
            print(f"Error in speaker_separation: {e}")
            print(traceback.format_exc())
            return []

    def transcribe_segment(self, audio_path, video_path, audio_extract=True):
        """
        音声ファイルを処理し、無音区間の外側を保持して文字起こしを行う。

        Parameters:
            audio_path (str): 入力音声ファイルのパス。
            video_path (str): 入力動画ファイルのパス。
            audio_extract (bool): True の場合はボーカル抽出を行う。

        Returns:
            list: 文字起こし結果（各要素は dict 形式）。
        """
        print(f"Transcribing audio file: {audio_path}")

        # --- 話者分離を実行 ---
        print("Running speaker separation...")
        speaker_segments = self.speaker_separation(audio_path, video_path)
        print(f"Speaker separation completed. Found {len(speaker_segments)} segments.")

        # --- 各話者区間に対して無音検出と文字起こしを実行 ---
        all_transcriptions = []
        for segment in speaker_segments:
            segment_path = segment["audio_path"]
            speaker_id = segment["speaker"]

            print(f"Processing segment: {segment_path} (Speaker: {speaker_id})")

            # サンプリングレートを取得
            forced_sr = 16000
            audio_data, sr = librosa.load(segment_path, sr=forced_sr, mono=False)
            data_len_seconds = len(audio_data) / sr

            # 無音区間を検出
            silences = self.SileroVAD_detect_silence(
                audio_file=segment_path,
                sampling_rate=forced_sr,
                threshold=0.5
            )

            # 無音区間の外側を保持するブロックを計算
            keep_blocks = self.get_keep_blocks(
                silences=silences,
                data_len=data_len_seconds,
                padding_time=0.2
            )
            print(f"Keep blocks for speaker {speaker_id}: {len(keep_blocks)}")

            # 音声セグメントを保存
            segment_files = self.save_audio_segments(
                audio_data=audio_data,
                keep_blocks=keep_blocks,
                samplerate=sr
            )
            print(f"Segment files for speaker {speaker_id}: {len(segment_files)}")

            # 各セグメントを文字起こし
            transcriptions = self.transcribe_audio_and_split_video(segment_files, keep_blocks)
            for transcription in transcriptions:
                transcription["speaker"] = speaker_id  # 話者情報を追加
                all_transcriptions.append(transcription)

        return all_transcriptions
