import os
import torch
import torchaudio
from torchaudio.transforms import MFCC
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from tqdm import tqdm
import numpy as np
from transformers import pipeline
import librosa
import pandas as pd  # インポートを先頭に移動
import logging  # ロギングのために追加

# ロギングの設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class AudioComparator:
    def __init__(self, sampling_rate=16000, n_mfcc=13):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.sampling_rate = sampling_rate
        self.mfcc_transform = MFCC(
            sample_rate=sampling_rate,
            n_mfcc=n_mfcc
        ).to(self.device)

        # Silero VAD model loading once
        try:
            self.vad_model, self.vad_utils = torch.hub.load(
                repo_or_dir='snakers4/silero-vad',
                model='silero_vad',
                force_reload=False
            )
            self.get_speech_timestamps, self.save_audio, self.read_audio, _, _ = self.vad_utils
        except Exception as e:
            logging.error(f"Silero VADモデルの読み込み中にエラーが発生しました: {e}")
            raise

        # Kotoba-Whisper model pipeline
        try:
            self.kotoba_whisper_model_id = "kotoba-tech/kotoba-whisper-v1.0"
            self.kotoba_whisper_torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
            self.kotoba_whisper_device = 0 if torch.cuda.is_available() else -1  # デバイスIDに変更
            self.kotoba_whisper_model_kwargs = {"attn_implementation": "sdpa"} if torch.cuda.is_available() else {}
            self.kotoba_whisper_pipe = pipeline(
                "automatic-speech-recognition",
                model=self.kotoba_whisper_model_id,
                torch_dtype=self.kotoba_whisper_torch_dtype,
                device=self.kotoba_whisper_device,
                model_kwargs=self.kotoba_whisper_model_kwargs
            )
        except Exception as e:
            logging.error(f"Kotoba-Whisperモデルの読み込み中にエラーが発生しました: {e}")
            raise

    def SileroVAD_detect_silence(self, audio_file, threshold=0.5):
        """Silero VADを使用して音声ファイルの無音部分を検出します。"""
        try:
            if not os.path.exists(audio_file):
                raise FileNotFoundError(f"音声ファイルが見つかりません: {audio_file}")

            audio_tensor = self.read_audio(audio_file, sampling_rate=self.sampling_rate)

            speech_timestamps = self.get_speech_timestamps(
                audio_tensor, self.vad_model, threshold=threshold, sampling_rate=self.sampling_rate
            )

            if not speech_timestamps:
                logging.info(f"{audio_file}で音声が検出されませんでした。")
                return []

            silences = []
            last_end = 0
            audio_length = audio_tensor.shape[-1]
            for segment in speech_timestamps:
                if last_end < segment['start']:
                    silences.append({
                        "from": last_end / self.sampling_rate,
                        "to": segment['start'] / self.sampling_rate,
                        "suffix": "cut"
                    })
                last_end = segment['end']

            if last_end < audio_length:
                silences.append({
                    "from": last_end / self.sampling_rate,
                    "to": audio_length / self.sampling_rate,
                    "suffix": "cut"
                })

            return silences
        except Exception as e:
            logging.error(f"SileroVAD_detect_silence中でエラーが発生しました: {e}")
            return []

    def compute_full_mfcc(self, audio_path):
        """音声ファイル全体のMFCCを計算します。"""
        try:
            waveform, file_sr = torchaudio.load(audio_path)

            if file_sr != self.sampling_rate:
                resampler = torchaudio.transforms.Resample(orig_freq=file_sr, new_freq=self.sampling_rate).to(
                    self.device)
                waveform = waveform.to(self.device)
                waveform = resampler(waveform)
            else:
                waveform = waveform.to(self.device)

            mfcc = self.mfcc_transform(waveform)
            return mfcc.squeeze(0)

        except torch.cuda.OutOfMemoryError as e:
            logging.error("CUDAメモリ不足エラー:", exc_info=True)
            torch.cuda.empty_cache()
            raise
        except Exception as e:
            logging.error("MFCC計算中にエラーが発生しました:", exc_info=True)
            raise

    def extract_mfcc_block(self, full_mfcc, start, end):
        """特定の時間ブロックのMFCCを抽出します。"""
        hop_length = 512  # デフォルトのホップ長
        start_frame = int(start * self.sampling_rate / hop_length)
        end_frame = int(end * self.sampling_rate / hop_length)
        block_mfcc = full_mfcc[:, start_frame:end_frame]
        return block_mfcc.mean(dim=-1).cpu().numpy()

    def kotoba_whisper(self, audio_file, max_segment_duration=30.0):
        """
        kotoba-Whisperを使用して音声を文字起こしします。

        Parameters:
            audio_file (str): 入力音声ファイルのパス。
            max_segment_duration (float): 最大セグメント長（秒）。

        Returns:
            str: 文字起こし結果。
        """
        generate_kwargs = {
            "return_timestamps": True,
            "language": "japanese"
        }

        try:
            audio, sampling_rate = librosa.load(audio_file, sr=self.sampling_rate)
            total_duration = len(audio) / self.sampling_rate
            segments = []

            # 音声をセグメントに分割
            for start in np.arange(0, total_duration, max_segment_duration):
                end = min(start + max_segment_duration, total_duration)
                start_sample = int(start * self.sampling_rate)
                end_sample = int(end * self.sampling_rate)
                segments.append(audio[start_sample:end_sample])

            # 各セグメントを処理
            transcriptions = []
            for segment in segments:
                audio_input = {"raw": segment, "sampling_rate": self.sampling_rate}
                result = self.kotoba_whisper_pipe(audio_input, generate_kwargs=generate_kwargs)
                transcriptions.append(result["text"])

            # セグメントの文字起こしを結合
            return " ".join(transcriptions)
        except Exception as e:
            logging.error(f"kotoba_whisper中でエラーが発生しました: {e}", exc_info=True)
            return ""

    def transcribe_blocks(self, audio_file, blocks):
        """
        kotoba-whisperを使用して音声ブロックを文字起こしします。

        Parameters:
            audio_file (str): 音声ファイルのパス。
            blocks (list): "from" と "to" 時間を含むブロックのリスト。

        Returns:
            list: 各ブロックの文字起こし結果のリスト。
        """
        transcriptions = []
        try:
            audio, sr = librosa.load(audio_file, sr=self.sampling_rate)
            for block in tqdm(blocks, desc="Transcribing Blocks"):
                start_sample = int(block["from"] * sr)
                end_sample = int(block["to"] * sr)
                block_audio = audio[start_sample:end_sample]

                audio_input = {"raw": block_audio, "sampling_rate": sr}
                result = self.kotoba_whisper_pipe(audio_input,
                                                  generate_kwargs={"return_timestamps": True, "language": "japanese"})

                transcriptions.append({
                    "text": result["text"],
                    "start": block["from"],
                    "end": block["to"]
                })
        except Exception as e:
            logging.error(f"transcribe_blocks中でエラーが発生しました: {e}", exc_info=True)
        return transcriptions

    def calculate_audio_statistics(self, audio_file, blocks):
        """
        音声の統計情報（全体の平均音量と各ブロックの分散）を計算します。

        Parameters:
            audio_file (str): 音声ファイルのパス。
            blocks (list): "from" と "to" 時間を含むブロックのリスト。

        Returns:
            dict: 全体の平均音量と各ブロックの分散。
        """
        try:
            audio, sr = librosa.load(audio_file, sr=self.sampling_rate)
            overall_mean = np.mean(np.abs(audio))

            block_statistics = []
            for block in blocks:
                start_sample = int(block["from"] * sr)
                end_sample = int(block["to"] * sr)
                block_audio = audio[start_sample:end_sample]
                block_variance = np.var(block_audio)
                block_statistics.append({
                    "from": block["from"],
                    "to": block["to"],
                    "variance": block_variance
                })

            return {
                "overall_mean": overall_mean,
                "block_statistics": block_statistics
            }
        except Exception as e:
            logging.error(f"calculate_audio_statistics中でエラーが発生しました: {e}", exc_info=True)
            return {}

    def compare_audio_blocks(self, source_audio, clipping_audio, source_blocks, clipping_blocks, initial_threshold=100,
                             threshold_increment=50):
        """ソース音声とクリッピング音声の各ブロックを比較します。"""
        try:
            source_full_mfcc = self.compute_full_mfcc(source_audio)
            clipping_full_mfcc = self.compute_full_mfcc(clipping_audio)

            matches = []
            current_threshold = initial_threshold
            source_index = 0

            for j in tqdm(range(len(clipping_blocks)), desc="Processing Blocks"):
                clip_block = clipping_blocks[j]
                clip_mfcc = self.extract_mfcc_block(clipping_full_mfcc, clip_block["from"], clip_block["to"])
                match_found = False

                while not match_found:
                    for i in range(source_index, len(source_blocks)):
                        source_block = source_blocks[i]
                        source_mfcc = self.extract_mfcc_block(source_full_mfcc, source_block["from"],
                                                              source_block["to"])

                        distance = fastdtw(clip_mfcc, source_mfcc, dist=euclidean)[0]

                        if distance < current_threshold:
                            matches.append({
                                "clip_start": clip_block["from"],
                                "clip_end": clip_block["to"],
                                "source_start": source_block["from"],
                                "source_end": source_block["to"],
                                "distance": distance  # distanceを追加
                            })
                            source_index = i + 1
                            match_found = True
                            break

                    if not match_found:
                        current_threshold += threshold_increment
                        if current_threshold > 1000:
                            logging.warning(f"クリップブロック {j} のマッチが見つかりませんでした。閾値を {current_threshold} まで上げました。")
                            break

            # 必ず2つの値を返す
            return matches, current_threshold
        except Exception as e:
            logging.error(f"compare_audio_blocks中でエラーが発生しました: {e}", exc_info=True)
            return [], initial_threshold

    def process_audio(self, source_audio, clipping_audio):
        """
        ソース音声とクリッピング音声を処理して、マッチするブロックを見つけて文字起こしします。

        Parameters:
            source_audio (str): ソース音声ファイルのパス。
            clipping_audio (str): クリッピング音声ファイルのパス。

        Returns:
            dict: マッチしたブロックと最終的な閾値を含む詳細な結果。
        """
        try:
            source_blocks = self.SileroVAD_detect_silence(source_audio)
            clipping_blocks = self.SileroVAD_detect_silence(clipping_audio)

            if not source_blocks:
                logging.info("ソース音声で音声ブロックが検出されませんでした。")
                return {"blocks": [], "current_threshold": 100}

            if not clipping_blocks:
                logging.info("クリッピング音声で音声ブロックが検出されませんでした。")
                return {"blocks": [], "current_threshold": 100}

            transcriptions = self.transcribe_blocks(source_audio, source_blocks)

            # compare_audio_blocksの戻り値を確実に取得
            matches, final_threshold = self.compare_audio_blocks(source_audio, clipping_audio, source_blocks,
                                                                 clipping_blocks)

            matched_indices = {match["source_start"]: match for match in matches}
            detailed_results = []

            for source_block in source_blocks:
                match = matched_indices.get(source_block["from"])
                detailed_results.append({
                    "clip_start": match["clip_start"] if match else None,
                    "clip_end": match["clip_end"] if match else None,
                    "source_start": source_block["from"],
                    "source_end": source_block["to"],
                    "text": next((t["text"] for t in transcriptions if t["start"] == source_block["from"]), ""),
                    "matched": bool(match),
                    "distance": match["distance"] if match else None  # distanceを追加
                })

            return {
                "blocks": detailed_results,
                "current_threshold": final_threshold
            }
        except Exception as e:
            logging.error(f"process_audio中でエラーが発生しました: {e}", exc_info=True)
            return {"blocks": [], "current_threshold": 100}


if __name__ == "__main__":
    # パスを適切に設定してください
    source_audio = "../data/audio/source/bh4ObBry9q4.mp3"
    clipping_audio = "../data/audio/clipping/84iD1TEttV0.mp3"

    comparator = AudioComparator(sampling_rate=16000)

    # 音声を処理して結果を取得
    result = comparator.process_audio(source_audio, clipping_audio)

    if result["blocks"]:
        # Pandas DataFrame に保存
        output_csv = "transcription_results_with_match.csv"
        df = pd.DataFrame(result["blocks"])
        df.to_csv(output_csv, index=False, encoding='utf-8-sig')

        logging.info(f"文字起こし結果が {output_csv} に保存されました。")

        print("Transcriptions:")
        for block in result["blocks"]:
            print(f"Clip Start: {block['clip_start']}, Clip End: {block['clip_end']}, "
                  f"Source Start: {block['source_start']}, Source End: {block['source_end']}, "
                  f"Matched: {block['matched']}, Distance: {block['distance']}, Text: {block['text']}")
    else:
        logging.info("マッチするブロックが見つからないか、文字起こしに失敗しました。")

    logging.info(f"\n最終的な閾値: {result['current_threshold']}")
