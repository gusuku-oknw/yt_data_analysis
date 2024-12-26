import subprocess
import traceback
import torch
import librosa
import os
import logging
from tqdm import tqdm
from datetime import timedelta
from typing import List, Tuple, Dict, Optional, Union
from transformers import Wav2Vec2Processor, Wav2Vec2Model

# ログ設定
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

class Wav2Vec2Analyzer:
    def __init__(self, model_name: str = "facebook/wav2vec2-large-960h-lv60-self", sampling_rate: int = 16000):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.sampling_rate = sampling_rate

        # モデルとプロセッサのロード
        self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        self.model = Wav2Vec2Model.from_pretrained(model_name).to(self.device)

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

            logging.info(f"Detected silences: {silences}")
            return silences

        except Exception as e:
            logging.error(f"Error detecting silence: {e}")
            traceback.print_exc()
            return []

    def split_audio(self, audio_tensor: torch.Tensor, silences: List[Dict[str, float]], min_length: float = 5.0) -> \
    Tuple[List[torch.Tensor], List[Tuple[float, float]]]:
        """無音区間で音声を分割し、最小長を満たさない場合はセグメントを削除する"""
        segments = []
        segment_times = []
        buffer = torch.tensor([])  # バッファ用のテンソル
        buffer_start = 0  # バッファの開始時間

        last_end = 0
        for silence in silences:
            start_sample = int(last_end * self.sampling_rate)
            end_sample = int(silence["from"] * self.sampling_rate)

            # 現在のセグメントをバッファに追加
            current_segment = audio_tensor[start_sample:end_sample]
            buffer = torch.cat((buffer, current_segment))

            # バッファの長さが最小長を超えた場合
            if len(buffer) / self.sampling_rate >= min_length:
                segments.append(buffer)
                segment_times.append((buffer_start, silence["from"]))
                buffer = torch.tensor([])  # バッファをリセット
                buffer_start = silence["to"]  # 次のバッファの開始時間を更新

            last_end = silence["to"]

        # 最後のバッファが残っている場合
        if len(buffer) / self.sampling_rate >= min_length:
            segments.append(buffer)
            segment_times.append((buffer_start, last_end))

        logging.info(f"Split audio into {len(segments)} segments.")
        return segments, segment_times

    def calculate_similarity(self, segment1: torch.Tensor, segment2: torch.Tensor) -> float:
        """音声セグメント間のコサイン類似度を計算"""
        inputs = self.processor(
            [segment1.numpy(), segment2.numpy()],  # Convert tensors to numpy arrays
            sampling_rate=self.sampling_rate,
            return_tensors="pt",
            padding=True  # Ensure padding is enabled
        )
        inputs = {key: tensor.to(self.device) for key, tensor in inputs.items()}

        with torch.no_grad():
            embeddings = self.model(**inputs).last_hidden_state.mean(dim=1)

        embeddings = torch.nn.functional.normalize(embeddings, dim=-1)
        cosine_sim = torch.nn.CosineSimilarity(dim=-1)

        similarity = cosine_sim(embeddings[0], embeddings[1]).item()
        logging.debug(f"Calculated similarity: {similarity}")
        return similarity

    def find_closest_segments(self, clip: torch.Tensor, source: torch.Tensor, similarity_threshold: float = 0.8) -> List[Tuple[int, int, float, Tuple[float, float], Tuple[float, float]]]:
        """2つの音声間で最も類似したセグメントを検索"""
        silences1 = self.detect_silence(clip, threshold=0.4)
        silences2 = self.detect_silence(source)

        segments1, times1 = self.split_audio(clip, silences1)
        segments2, times2 = self.split_audio(source, silences2)

        closest_pairs = []
        for i, (seg2, time2) in tqdm(enumerate(zip(segments2, times2)), total=len(segments2)):
            best_match = None
            best_similarity = -1
            best_time = None

            for j, (seg1, time1) in enumerate(zip(segments1, times1)):
                similarity = self.calculate_similarity(seg1, seg2)
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match = j
                    best_time = time1

            if best_similarity >= similarity_threshold:
                closest_pairs.append((i, best_match, best_similarity, time2, best_time))

        logging.info(f"Found {len(closest_pairs)} closest segment pairs.")
        return closest_pairs

    def format_time(self, seconds: float) -> str:
        """秒を時間:分:秒にフォーマット"""
        return str(timedelta(seconds=round(seconds)))

if __name__ == "__main__":
    source_audio_file = "../data/audio/source/pnHdRQbR2zs.mp3"
    clip_audio_file = "../data/audio/clipping/-bRcKCM5_3E.mp3"

    analyzer = Wav2Vec2Analyzer()

    source_audio, _ = librosa.load(source_audio_file, sr=analyzer.sampling_rate)
    clip_audio, _ = librosa.load(clip_audio_file, sr=analyzer.sampling_rate)

    closest_segments = analyzer.find_closest_segments(torch.tensor(clip_audio), torch.tensor(source_audio))

    for seg2_idx, seg1_idx, similarity, time2, time1 in closest_segments:
        print(f"Clip Segment {seg2_idx} (time: {analyzer.format_time(time2[0])}-{analyzer.format_time(time2[1])}) is closest to Source Segment {seg1_idx} (time: {analyzer.format_time(time1[0])}-{analyzer.format_time(time1[1])}) with similarity {similarity:.4f}")
