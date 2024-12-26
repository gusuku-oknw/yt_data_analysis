import subprocess
import traceback
import torch
import librosa
import os
import logging
from tqdm import tqdm
from datetime import timedelta
from typing import List, Tuple, Dict, Optional, Union
from transformers import Wav2Vec2FeatureExtractor, WavLMForXVector
from scipy.spatial.distance import euclidean
import numpy as np

# ログ設定
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s [%(levelname)s] %(message)s")

def dtw(x: np.ndarray, y: np.ndarray) -> float:
    """Compute the Dynamic Time Warping (DTW) distance between two 1-D sequences."""
    # Initialize the cost matrix with infinity
    cost = np.full((len(x) + 1, len(y) + 1), np.inf)
    cost[0, 0] = 0

    # Populate the cost matrix
    for i in range(1, len(x) + 1):
        for j in range(1, len(y) + 1):
            dist = abs(x[i - 1] - y[j - 1])
            cost[i, j] = dist + min(cost[i - 1, j],    # Insertion
                                    cost[i, j - 1],    # Deletion
                                    cost[i - 1, j - 1]) # Match

    # The DTW distance is the cost at the bottom-right corner of the matrix
    return cost[len(x), len(y)]

class WavLMAnalyzer:
    def __init__(self, model_name: str = "microsoft/wavlm-base-plus-sv", sampling_rate: int = 16000):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.sampling_rate = sampling_rate

        # モデルと特徴抽出器のロード
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
        self.model = WavLMForXVector.from_pretrained(model_name).to(self.device)

    def detect_silence(self, audio_input: Union[str, torch.Tensor], threshold: float = 0.4) -> List[Dict[str, float]]:
        """Silero VAD を用いて無音区間を検出"""
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

            # 無音区間を検出
            speech_timestamps = get_speech_timestamps(
                audio_tensor, model,
                threshold=threshold,
                sampling_rate=self.sampling_rate
            )

            silences = []
            last_end = 0
            for segment in speech_timestamps:
                if last_end < segment['start']:
                    silences.append({
                        "from": last_end / self.sampling_rate,
                        "to": segment['start'] / self.sampling_rate
                    })
                last_end = segment['end']

            if last_end < len(audio_tensor):
                silences.append({
                    "from": last_end / self.sampling_rate,
                    "to": len(audio_tensor) / self.sampling_rate
                })

            return silences

        except Exception as e:
            logging.error(f"Error detecting silence: {e}")
            traceback.print_exc()
            return []

    def split_audio(self, audio_tensor: torch.Tensor, silences: List[Dict[str, float]], min_length: float = 3.0) -> \
    Tuple[List[torch.Tensor], List[Tuple[float, float]]]:
        """無音区間で音声を分割し、最小長を満たさない場合は次のセグメントと合成する"""
        segments = []
        segment_times = []
        buffer = torch.empty(0, dtype=audio_tensor.dtype, device=audio_tensor.device)  # Initialize buffer correctly
        buffer_start = 0  # Buffer start time

        last_end = 0
        for silence in silences:
            start_sample = int(last_end * self.sampling_rate)
            end_sample = int(silence["from"] * self.sampling_rate)

            # Current segment to add to buffer
            current_segment = audio_tensor[start_sample:end_sample]
            buffer = torch.cat((buffer, current_segment))

            # If buffer length exceeds minimum length, add to segments
            if len(buffer) / self.sampling_rate >= min_length:
                if len(buffer) > 0:
                    segments.append(buffer.clone())
                    segment_times.append((buffer_start, silence["from"]))
                    logging.debug(f"Added segment: {buffer_start}-{silence['from']} seconds, length: {len(buffer)} samples")
                buffer = torch.empty(0, dtype=audio_tensor.dtype, device=audio_tensor.device)  # Reset buffer
                buffer_start = silence["to"]  # Update buffer start time

            last_end = silence["to"]

        # Handle the last buffer
        if len(buffer) > 0:
            if len(buffer) / self.sampling_rate >= min_length:
                segments.append(buffer.clone())
                segment_times.append((buffer_start, last_end))
                logging.debug(f"Added final segment: {buffer_start}-{last_end} seconds, length: {len(buffer)} samples")
            else:
                # Merge with the last segment if it doesn't meet the minimum length
                if len(segments) > 0:
                    segments[-1] = torch.cat((segments[-1], buffer))
                    segment_times[-1] = (segment_times[-1][0], last_end)
                    logging.debug(f"Merged segment: {segment_times[-1][0]}-{last_end} seconds")

        # Log the number of segments created
        logging.info(f"Total segments created: {len(segments)}")

        return segments, segment_times

    def calculate_similarity(self, segment1: torch.Tensor, segment2: torch.Tensor) -> float:
        """Calculate similarity between two audio segments using DTW distance."""
        # Ensure tensors are on CPU and converted to NumPy arrays
        segment1_np = segment1.cpu().detach().numpy().flatten()
        segment2_np = segment2.cpu().detach().numpy().flatten()

        # Compute the DTW distance
        dtw_distance = dtw(segment1_np, segment2_np)

        # Convert distance to similarity (the smaller the distance, the higher the similarity)
        similarity = 1 / (1 + dtw_distance)
        return similarity

    def find_closest_segments(
        self,
        audio1: torch.Tensor,
        audio2: torch.Tensor,
        similarity_threshold: float = 0.9,
        max_segments_to_process: int = 10  # New parameter for limiting segments
    ) -> List[
        Tuple[int, int, float, Tuple[float, float], Tuple[float, float]]
    ]:
        """Find the most similar segments between two audio files.

        Parameters
        ----------
        audio1 : torch.Tensor
            比較対象の音声データ1
        audio2 : torch.Tensor
            比較対象の音声データ2
        similarity_threshold : float
            類似度の閾値。この値以上のセグメントペアのみを返す
        max_segments_to_process : int
            For testing purposes, limit the number of segments processed.

        Returns
        -------
        List[Tuple[int, int, float, Tuple[float, float], Tuple[float, float]]]
            類似セグメントのペアのリスト。各要素は以下を含む:
            - セグメント1のインデックス
            - セグメント2のインデックス
            - 類似度スコア
            - セグメント1の開始・終了時間
            - セグメント2の開始・終了時間
        """
        # Detect silences
        silences1 = self.detect_silence(audio1)
        silences2 = self.detect_silence(audio2)

        # Split audio based on silences
        segments1, times1 = self.split_audio(audio1, silences1)
        segments2, times2 = self.split_audio(audio2, silences2)

        closest_pairs = []

        # Process only a limited number of segments for testing
        for i, (seg2, time2) in tqdm(enumerate(zip(segments2, times2)), total=min(len(segments2), max_segments_to_process)):
            if i >= max_segments_to_process:
                break
            best_match = None
            best_similarity = -1
            best_time = None

            # Compare with all segments in audio1
            for j, (seg1, time1) in enumerate(zip(segments1, times1)):
                similarity = self.calculate_similarity(seg1, seg2)
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match = j
                    best_time = time1

            # Save pairs that exceed the similarity threshold
            if best_similarity >= similarity_threshold:
                closest_pairs.append((best_match, i, best_similarity, best_time, time2))

        return closest_pairs

    def extract_vocals(self, audio_file: str) -> Optional[str]:
        """Demucs を用いてボーカルを抽出"""
        try:
            root_directory = os.path.dirname(audio_file)
            basename = os.path.splitext(os.path.basename(audio_file))[0]
            vocals_path = os.path.join(root_directory, "htdemucs", basename, "vocals.wav")

            if os.path.exists(vocals_path):
                logging.info(f"Vocals already extracted: {vocals_path}")
                return vocals_path

            device = "cuda" if torch.cuda.is_available() else "cpu"
            command = ['demucs', '-d', device, '-o', root_directory, audio_file]
            subprocess.run(command, check=True)

            if os.path.exists(vocals_path):
                logging.info(f"Vocals extracted: {vocals_path}")
                return vocals_path

            raise FileNotFoundError(f"Vocals not found after Demucs execution: {vocals_path}")

        except subprocess.CalledProcessError as e:
            logging.error(f"Error executing Demucs: {e}")
        except Exception as e:
            logging.error(f"Unexpected error during vocal extraction: {e}")

        return None

    def format_time(self, seconds: float) -> str:
        """秒を時間:分:秒にフォーマット"""
        return str(timedelta(seconds=round(seconds)))

if __name__ == "__main__":
    # 音声ファイルの設定
    source_audio_file = "../data/audio/source/pnHdRQbR2zs.mp3"
    clip_audio_file = "../data/audio/clipping/-bRcKCM5_3E.mp3"

    analyzer = WavLMAnalyzer()

    # ボーカル抽出 (Optional)
    # clip_audio_file = analyzer.extract_vocals(clip_audio_file)

    # 音声ファイルをロード
    logging.info(f"Loading source audio from {source_audio_file}")
    source_audio, _ = librosa.load(source_audio_file, sr=analyzer.sampling_rate)
    logging.info(f"Source audio loaded with {len(source_audio)} samples.")

    logging.info(f"Loading clip audio from {clip_audio_file}")
    clip_audio, _ = librosa.load(clip_audio_file, sr=analyzer.sampling_rate)
    logging.info(f"Clip audio loaded with {len(clip_audio)} samples.")

    # 再度モノラルチェック (冗長だが安全のため)
    if source_audio.ndim > 1:
        source_audio = librosa.to_mono(source_audio)
        logging.info("Converted source audio to mono.")
    else:
        logging.info("Source audio is already mono.")

    if clip_audio.ndim > 1:
        clip_audio = librosa.to_mono(clip_audio)
        logging.info("Converted clip audio to mono.")
    else:
        logging.info("Clip audio is already mono.")

    # Convert to torch tensors with appropriate dtype
    source_tensor = torch.tensor(source_audio, dtype=torch.float32)
    clip_tensor = torch.tensor(clip_audio, dtype=torch.float32)

    # 類似セグメントを検索
    logging.info("Finding closest segments between source and clip audio.")
    closest_segments = analyzer.find_closest_segments(clip_tensor, source_tensor)

    # 結果を出力
    if closest_segments:
        logging.info(f"Found {len(closest_segments)} similar segments.")
        for seg1_idx, seg2_idx, similarity, time1, time2 in closest_segments:
            print(
                f"Clip Segment {seg1_idx} (time: {analyzer.format_time(time1[0])}-{analyzer.format_time(time1[1])}) "
                f"is closest to Source Segment {seg2_idx} (time: {analyzer.format_time(time2[0])}-{analyzer.format_time(time2[1])}) "
                f"with similarity {similarity:.4f}"
            )
    else:
        logging.info("No similar segments found.")
