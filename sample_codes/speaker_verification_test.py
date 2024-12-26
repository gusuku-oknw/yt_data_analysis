import os
import torch
import logging
import traceback
from itertools import product
from typing import List, Union
from speechbrain.inference.speaker import EncoderClassifier
import torchaudio
from datetime import timedelta


def format_time(seconds: float) -> str:
    """秒を時間:分:秒にフォーマット"""
    return str(timedelta(seconds=round(seconds)))


class SpeechSegmentComparison:
    def __init__(self, sampling_rate: int = 16000, threshold: float = 0.95, model_name="speechbrain/spkrec-ecapa-voxceleb"):
        self.sampling_rate = sampling_rate
        self.threshold = threshold
        self.recognizer = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")

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

    def extract_embedding(self, audio_path: str) -> torch.Tensor:
        """
        音声ファイルから話者埋め込みを抽出する。

        Parameters:
        -----------
        audio_path : str
            音声ファイルのパス

        Returns:
        --------
        torch.Tensor
            音声埋め込みベクトル
        """
        signal, _ = torchaudio.load(audio_path)
        embeddings = self.recognizer.encode_batch(signal)
        return embeddings

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
                for segment2 in speech_segments2:
                    # print(f"Segment1: {format_time(segment1['from'])} - {format_time(segment1['to'])}")
                    # print(f"Segment2: {format_time(segment2['from'])} - {format_time(segment2['to'])}")

                    # Dummy distance calculation (replace with real calculation as needed)
                    distance = abs((segment1['to'] - segment1['from']) - (segment2['to'] - segment2['from']))

                    # print(f"Distance: {distance}")
                    if distance < best_score:
                        best_score = distance
                        best_segment = segment2

                if best_segment:
                    print(f"Best match for Segment1 {format_time(segment1['from'])} - {format_time(segment1['to'])} is Segment2 {format_time(best_segment['from'])} - {format_time(best_segment['to'])} with Score: {best_score}\n")

        except Exception as e:
            logging.error(f"Error comparing segments: {e}")
            traceback.print_exc()

if __name__ == "__main__":
    source_audio_file = "../data/audio/source/pnHdRQbR2zs.mp3"
    clip_audio_file = "../data/audio/clipping/-bRcKCM5_3E.mp3"
    clip_audio_file2 = "../data/audio/clipping/htdemucs/-bRcKCM5_3E/vocals.wav"

    comparator = SpeechSegmentComparison()
    comparator.compare_segments(clip_audio_file2, source_audio_file)
