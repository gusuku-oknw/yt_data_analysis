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

# ログ設定
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

class WavLMAnalyzer:
    def __init__(self, model_name: str = "microsoft/wavlm-base-plus-sv", sampling_rate: int = 16000):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.sampling_rate = sampling_rate

        # モデルと特徴抽出器のロード
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
        self.model = WavLMForXVector.from_pretrained(model_name).to(self.device)

    def detect_silence(self, audio_input: Union[str, torch.Tensor], threshold: float = 0.6) -> List[Dict[str, float]]:
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

    def split_audio(self, audio_tensor: torch.Tensor, silences: List[Dict[str, float]], min_length: float = 5.0) -> \
    Tuple[List[torch.Tensor], List[Tuple[float, float]]]:
        """無音区間で音声を分割し、最小長を満たさない場合は次のセグメントと合成する"""
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
        if len(buffer) > 0:
            if len(buffer) / self.sampling_rate >= min_length:
                segments.append(buffer)
                segment_times.append((buffer_start, last_end))
            else:
                # 最後のバッファが最小長に満たない場合、前のセグメントに結合
                if len(segments) > 0:
                    segments[-1] = torch.cat((segments[-1], buffer))
                    segment_times[-1] = (segment_times[-1][0], last_end)

        return segments, segment_times

    def calculate_similarity(self, segment1: torch.Tensor, segment2: torch.Tensor) -> float:
        """音声セグメント間のコサイン類似度を計算"""
        inputs = self.feature_extractor(
            [segment1.numpy(), segment2.numpy()],  # Convert tensors to numpy arrays
            sampling_rate=self.sampling_rate,
            return_tensors="pt",
            padding=True,  # Ensure padding is enabled
        )
        inputs = {key: tensor.to(self.device) for key, tensor in inputs.items()}

        with torch.no_grad():
            embeddings = self.model(**inputs).embeddings

        embeddings = torch.nn.functional.normalize(embeddings, dim=-1)
        cosine_sim = torch.nn.CosineSimilarity(dim=-1)

        return cosine_sim(embeddings[0], embeddings[1]).item()


    def find_closest_segments(self, clip: torch.Tensor, source: torch.Tensor, similarity_threshold: float = 0.9) -> List[Tuple[int, int, float, Tuple[float, float], Tuple[float, float]]]:
        """2つの音声間で最も類似したセグメントを検索

            Parameters
            ----------
            clip : torch.Tensor
                比較対象の音声データ1
            source : torch.Tensor
                比較対象の音声データ2
            similarity_threshold : float
                類似度の閾値。この値以上のセグメントペアのみを返す

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
        # 無音区間を検出
        silences1 = self.detect_silence(clip, threshold=0.4)
        silences2 = self.detect_silence(source)

        # 音声を無音区間で分割
        segments1, times1 = self.split_audio(clip, silences1)
        segments2, times2 = self.split_audio(source, silences2)

        closest_pairs = []
        # 各セグメントについて最も類似したペアを探索
        for i, (seg2, time2) in tqdm(enumerate(zip(segments2, times2))):
            best_match = None
            best_similarity = -1
            best_time = None

            # セグメント間の類似度を計算
            for j, (seg1, time1) in enumerate(zip(segments1, times1)):
                similarity = self.calculate_similarity(seg1, seg2)
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match = j
                    best_time = time1

            # 閾値以上の類似度を持つペアのみを保存
            if best_similarity >= similarity_threshold:
                closest_pairs.append((i, best_match, best_similarity, time2, best_time))

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

    # ボーカル抽出
    # clip_audio_file = analyzer.extract_vocals(clip_audio_file)

    # 音声ファイルをロード
    source_audio, _ = librosa.load(source_audio_file, sr=analyzer.sampling_rate)
    clip_audio, _ = librosa.load(clip_audio_file, sr=analyzer.sampling_rate)

    # 類似セグメントを検索
    closest_segments = analyzer.find_closest_segments(torch.tensor(clip_audio), torch.tensor(source_audio))

    # 結果を出力
    for seg2_idx, seg1_idx, similarity, time2, time1 in closest_segments:
        print(f"Clip Segment {seg2_idx} (time: {analyzer.format_time(time2[0])}-{analyzer.format_time(time2[1])}) is closest to Source Segment {seg1_idx} (time: {analyzer.format_time(time1[0])}-{analyzer.format_time(time1[1])}) with similarity {similarity:.4f}")
