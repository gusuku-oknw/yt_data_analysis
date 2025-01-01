import os
import torch
import logging
import traceback
from itertools import product
from typing import List, Union
import torchaudio
import librosa
import numpy as np
from datetime import timedelta

def format_time(seconds: float) -> str:
    """秒を時間:分:秒にフォーマット"""
    return str(timedelta(seconds=round(seconds)))

class SpeechSegmentComparison:
    def __init__(self, sampling_rate: int = 16000, threshold: float = 0.7):
        self.sampling_rate = sampling_rate
        self.threshold = threshold

    def detect_speech(self, audio_input: Union[str, torch.Tensor]) -> List[dict]:
        """
        Silero VAD を用いて音声区間を検出する。

        Parameters:
        -----------
        audio_input : str | torch.Tensor
            音声ファイルのパスまたはテンソル形式の音声データ

        Returns:
        --------
        List[dict]
            音声区間のリスト
        """
        try:
            model, utils = torch.hub.load(
                repo_or_dir='snakers4/silero-vad',
                model='silero_vad',
                force_reload=True
            )
            (get_speech_timestamps, _, read_audio, _, _) = utils

            if isinstance(audio_input, str):
                if not os.path.exists(audio_input):
                    raise FileNotFoundError(f"Audio file not found: {audio_input}")
                audio_tensor = read_audio(audio_input, sampling_rate=self.sampling_rate)
            else:
                audio_tensor = audio_input

            # 音声区間を検出
            speech_timestamps = get_speech_timestamps(
                audio_tensor, model,
                threshold=self.threshold,
                sampling_rate=self.sampling_rate
            )

            speech_segments = []
            for segment in speech_timestamps:
                speech_segments.append({
                    "from": segment['start'] / self.sampling_rate,
                    "to": segment['end'] / self.sampling_rate
                })

            return speech_segments

        except Exception as e:
            logging.error(f"Error detecting speech: {e}")
            traceback.print_exc()
            return []

    def extract_spectrogram(self, audio_path: str, segment: dict) -> np.ndarray:
        """
        指定された音声セグメントからスペクトログラムを抽出する。

        Parameters:
        -----------
        audio_path : str
            音声ファイルのパス
        segment : dict
            セグメントの開始時間と終了時間

        Returns:
        --------
        np.ndarray
            スペクトログラム
        """
        signal, sr = librosa.load(audio_path, sr=self.sampling_rate, offset=segment['from'], duration=segment['to'] - segment['from'])
        spectrogram = np.abs(librosa.stft(signal))
        return spectrogram

    def pad_spectrogram(self, spec1: np.ndarray, spec2: np.ndarray) -> (np.ndarray, np.ndarray):
        """
        2つのスペクトログラムの形状をゼロ埋めで揃える。

        Parameters:
        -----------
        spec1 : np.ndarray
            スペクトログラム 1
        spec2 : np.ndarray
            スペクトログラム 2

        Returns:
        --------
        np.ndarray, np.ndarray
            サイズが揃った2つのスペクトログラム
        """
        max_rows = max(spec1.shape[0], spec2.shape[0])
        max_cols = max(spec1.shape[1], spec2.shape[1])

        # ランダムノイズを埋める
        noise_level = 1e-6  # 小さな値
        padded_spec1 = np.pad(
            spec1,
            ((0, max_rows - spec1.shape[0]), (0, max_cols - spec1.shape[1])),
            mode='constant',
            constant_values=noise_level
        )
        padded_spec2 = np.pad(
            spec2,
            ((0, max_rows - spec2.shape[0]), (0, max_cols - spec2.shape[1])),
            mode='constant',
            constant_values=noise_level
        )

        # 正規化してスコアの公平性を向上
        padded_spec1 /= np.linalg.norm(padded_spec1) + noise_level
        padded_spec2 /= np.linalg.norm(padded_spec2) + noise_level

        return padded_spec1, padded_spec2

    def compare_segments(self, audio_path1: str, audio_path2: str):
        """
        2つの音声ファイルをセグメントに分割し、総当たりで比較する。

        Parameters:
        -----------
        audio_path1 : str
            音声ファイル 1 のパス
        audio_path2 : str
            音声ファイル 2 のパス

        Returns:
        --------
        None
        """
        try:
            speech_segments1 = self.detect_speech(audio_path1)
            speech_segments2 = self.detect_speech(audio_path2)

            print("\nComparing speech segments:")
            for segment1 in speech_segments1:
                best_score = float('inf')
                best_segment = None

                spectrogram1 = self.extract_spectrogram(audio_path1, segment1)

                for segment2 in speech_segments2:
                    spectrogram2 = self.extract_spectrogram(audio_path2, segment2)

                    # スペクトログラムの形状を揃える
                    padded_spec1, padded_spec2 = self.pad_spectrogram(spectrogram1, spectrogram2)

                    # コサイン類似度を計算
                    similarity = np.dot(padded_spec1.flatten(), padded_spec2.flatten()) / (
                        np.linalg.norm(padded_spec1.flatten()) * np.linalg.norm(padded_spec2.flatten())
                    )

                    distance = 1 - similarity  # 類似度を距離に変換

                    if distance < best_score:
                        best_score = distance
                        best_segment = segment2

                if best_segment:
                    print(f"Best match for Segment1 {format_time(segment1['from'])} - {format_time(segment1['to'])} is Segment2 {format_time(best_segment['from'])} - {format_time(best_segment['to'])} with Distance: {best_score}\n")

        except Exception as e:
            logging.error(f"Error comparing segments: {e}")
            traceback.print_exc()

if __name__ == "__main__":
    source_audio_file = "../data/audio/source/pnHdRQbR2zs.mp3"
    clip_audio_file = "../data/audio/clipping/-bRcKCM5_3E.mp3"
    clip_audio_file2 = "../data/audio/clipping/htdemucs/-bRcKCM5_3E/vocals.wav"

    comparator = SpeechSegmentComparison()
    comparator.compare_segments(clip_audio_file2, source_audio_file)
