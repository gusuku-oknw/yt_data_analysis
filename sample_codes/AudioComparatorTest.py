import os
import torch
import torchaudio
from torchaudio.transforms import MFCC
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from tqdm import tqdm
import numpy as np
import logging

# ロギングの設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class AudioComparator:
    def __init__(self, sampling_rate, n_mfcc=20):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.sampling_rate = sampling_rate
        self.mfcc_transform = MFCC(
            sample_rate=sampling_rate,
            n_mfcc=n_mfcc
        ).to(self.device)

        # Silero VAD model loading
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

    def SileroVAD_detect_silence(self, audio_file, threshold=0.3):
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

    def compute_full_mfcc(self, audio_path, segment_duration=5.0):
        """
        Compute MFCC for audio in smaller segments to reduce memory usage.
        """
        torch.cuda.empty_cache()
        try:
            # Load the audio file
            waveform, file_sr = torchaudio.load(audio_path)

            # Resample if the sampling rate doesn't match
            if file_sr != self.sampling_rate:
                resampler = torchaudio.transforms.Resample(orig_freq=file_sr, new_freq=self.sampling_rate)
                waveform = resampler(waveform)

            # Move the waveform to the correct device
            waveform = waveform.to(self.device)

            # Calculate total duration
            total_duration = waveform.shape[1] / self.sampling_rate

            # Initialize variables
            segments = []

            # Process the waveform in smaller segments
            for start in np.arange(0, total_duration, segment_duration):
                end = min(start + segment_duration, total_duration)
                start_sample = int(start * self.sampling_rate)
                end_sample = int(end * self.sampling_rate)
                segment_waveform = waveform[:, start_sample:end_sample]

                if segment_waveform.shape[1] > 0:  # Ensure the segment is not empty
                    mfcc = self.mfcc_transform(segment_waveform).squeeze(0)  # Compute MFCC
                    segments.append(mfcc)
                    del segment_waveform, mfcc  # Free memory

            if not segments:
                raise RuntimeError("No valid segments were processed for MFCC computation.")

            # Determine the maximum length
            max_length = max(segment.size(1) for segment in segments)

            # Pad all segments to the maximum length
            padded_segments = []
            for idx, segment in enumerate(segments):
                padded_segment = torch.nn.functional.pad(segment, (0, max_length - segment.size(1)))
                padded_segments.append(padded_segment)

            # Concatenate all padded segments along the time axis
            full_mfcc = torch.cat(padded_segments, dim=2)

            # Clear CUDA memory cache
            torch.cuda.empty_cache()

            return full_mfcc

        except torch.cuda.OutOfMemoryError as e:
            logging.error("CUDA memory error during MFCC computation:", exc_info=True)
            torch.cuda.empty_cache()
            raise
        except Exception as e:
            logging.error("Error occurred during MFCC computation:", exc_info=True)
            raise

    def extract_mfcc_block(self, full_mfcc, start, end):
        """特定の時間ブロックのMFCCを抽出します。"""
        try:
            hop_length = 512  # デフォルトのホップ長
            start_frame = int(start * self.sampling_rate / hop_length)
            end_frame = int(end * self.sampling_rate / hop_length)
            block_mfcc = full_mfcc[:, :, start_frame:end_frame]  # 次元を確認

            if block_mfcc.size(2) == 0:  # セグメントが空の場合
                raise ValueError(f"Invalid block range: start={start}, end={end}, frames={start_frame}-{end_frame}")

            return block_mfcc.mean(dim=-1).cpu().numpy()
        except Exception as e:
            logging.error(f"extract_mfcc_block中でエラーが発生しました: {e}")
            raise

    def compare_audio_blocks(self, source_audio, clipping_audio, source_blocks, clipping_blocks, initial_threshold=10,
                             threshold_increment=25):
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
                logging.debug(f"Clip MFCC Shape: {clip_mfcc.shape}")

                match_found = False

                while not match_found:
                    for i in range(source_index, len(source_blocks)):
                        source_block = source_blocks[i]
                        source_mfcc = self.extract_mfcc_block(source_full_mfcc, source_block["from"],
                                                              source_block["to"])
                        logging.debug(f"Source MFCC Shape: {source_mfcc.shape}")

                        if clip_mfcc.shape[1] != source_mfcc.shape[1]:  # 次元が一致しない場合
                            raise ValueError(
                                f"Mismatch in MFCC dimensions: Clip({clip_mfcc.shape}) vs Source({source_mfcc.shape})")

                        distance = fastdtw(clip_mfcc, source_mfcc, dist=euclidean)[0]

                        if distance < current_threshold:
                            matches.append({
                                "clip_start": clip_block["from"],
                                "clip_end": clip_block["to"],
                                "source_start": source_block["from"],
                                "source_end": source_block["to"],
                                "distance": distance
                            })
                            source_index = i + 1
                            match_found = True
                            break

                    if not match_found:
                        current_threshold += threshold_increment
                        if current_threshold > 1000:
                            logging.warning(
                                f"クリップブロック {j} のマッチが見つかりませんでした。閾値を {current_threshold} まで上げました。")
                            break

            return matches, current_threshold
        except Exception as e:
            logging.error(f"compare_audio_blocks中でエラーが発生しました: {e}", exc_info=True)
            return [], initial_threshold

    def process_audio(self, source_audio, clipping_audio):
        """
        ソース音声とクリッピング音声を処理して、マッチするブロックを取得します。

        Parameters:
            source_audio (str): ソース音声ファイルのパス。
            clipping_audio (str): クリッピング音声ファイルのパス。

        Returns:
            list: マッチしたブロックの詳細情報を含むリスト。
        """
        try:
            source_blocks = self.SileroVAD_detect_silence(source_audio)
            clipping_blocks = self.SileroVAD_detect_silence(clipping_audio)

            if not source_blocks:
                logging.info("ソース音声で音声ブロックが検出されませんでした。")
                return []

            if not clipping_blocks:
                logging.info("クリッピング音声で音声ブロックが検出されませんでした。")
                return []

            matches, final_threshold = self.compare_audio_blocks(
                source_audio, clipping_audio, source_blocks, clipping_blocks
            )

            return matches
        except Exception as e:
            logging.error(f"process_audio中でエラーが発生しました: {e}", exc_info=True)
            return []

if __name__ == "__main__":
    source_audio = "../data/audio/source/bh4ObBry9q4.mp3"
    clipping_audio = "../data/audio/clipping/84iD1TEttV0.mp3"

    comparator = AudioComparator(sampling_rate=torchaudio.info(clipping_audio).sample_rate)
    result = comparator.process_audio(source_audio, clipping_audio)

    if result:
        print("マッチしたブロック:")
        for block in result:
            print(block)
    else:
        print("マッチするブロックが見つかりませんでした。")
