import os
import torch
import torchaudio
from torchaudio.transforms import MFCC
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from tqdm import tqdm

class AudioComparator:
    def __init__(self, sampling_rate=16000, n_mfcc=13):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.sampling_rate = sampling_rate
        self.mfcc_transform = MFCC(
            sample_rate=sampling_rate,
            n_mfcc=n_mfcc
        ).to(self.device)

    def SileroVAD_detect_silence(self, audio_file, threshold=0.5):
        """
        Silero VADを使って音声の区切り（静寂部分）を検出。
        """
        print(f"detect_silence: {audio_file}")
        try:
            # Silero VADモデルをロード
            model, utils = torch.hub.load(
                repo_or_dir='snakers4/silero-vad',
                model='silero_vad',
                force_reload=True
            )

            # ユーティリティ関数の取得
            (get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks) = utils

            # 音声ファイルを読み込み
            if not os.path.exists(audio_file):
                raise FileNotFoundError(f"Audio file not found: {audio_file}")

            audio_tensor = read_audio(audio_file, sampling_rate=self.sampling_rate)

            # 音声セグメントを検出
            speech_timestamps = get_speech_timestamps(
                audio_tensor, model, threshold=threshold, sampling_rate=self.sampling_rate
            )

            if not speech_timestamps:
                print(f"No speech detected in {audio_file}")
                return []

            # 静寂区間の計算
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
            print(f"An error occurred in SileroVAD_detect_silence: {e}")
            return []

    @staticmethod
    def extract_mfcc(audio_path, start, end, sr=16000, n_mfcc=13):
        """指定された区間のMFCCを抽出（1次元に変換）。"""
        y, _ = torchaudio.load(audio_path, frame_offset=int(start * sr), num_frames=int((end - start) * sr))
        waveform = y.to("cuda" if torch.cuda.is_available() else "cpu")

        # MFCCの計算
        mfcc = torchaudio.transforms.MFCC(sample_rate=sr, n_mfcc=n_mfcc).to(waveform.device)(waveform)

        # MFCCを2次元から1次元ベクトルに変換（時間軸方向の平均）
        mfcc_1d = mfcc.mean(dim=-1).squeeze(0).cpu().numpy()
        return mfcc_1d

    def compare_audio_blocks(self, source_audio, clipping_audio, source_blocks, clipping_blocks):
        """
        区切られたブロック同士で一致を確認し、逐次的に探索を最適化。
        """
        matches = []
        clip_index = 0  # 切り抜き動画の現在位置
        for i, source_block in tqdm(enumerate(source_blocks), total=len(source_blocks)):
            source_mfcc = self.extract_mfcc(
                source_audio, source_block["from"], source_block["to"]
            )
            # 逐次的に一致探索
            for j in range(clip_index, len(clipping_blocks)):
                clip_block = clipping_blocks[j]
                clip_mfcc = self.extract_mfcc(
                    clipping_audio, clip_block["from"], clip_block["to"]
                )
                # fastdtwに1次元ベクトルを渡す
                distance, _ = fastdtw(source_mfcc, clip_mfcc, dist=euclidean)

                if distance < 100:  # 閾値を調整
                    matches.append({
                        "source_start": source_block["from"],
                        "source_end": source_block["to"],
                        "clip_start": clip_block["from"],
                        "clip_end": clip_block["to"]
                    })
                    clip_index = j + 1  # 次の比較を現在の位置から開始
                    break  # 一度一致したら次のソースブロックへ

        # 結果としてstartとendだけを出力
        return [
            {
                "source": (match["source_start"], match["source_end"]),
                "clip": (match["clip_start"], match["clip_end"]),
            }
            for match in matches
        ]


if __name__ == "__main__":
    # ファイルパスの設定
    source_audio = "../data/audio/source/bh4ObBry9q4.mp3"
    clipping_audio = "../data/audio/clipping/84iD1TEttV0.mp3"

    # AudioComparatorのインスタンス作成
    comparator = AudioComparator()

    # VADを用いた音声区切りの取得
    source_blocks = comparator.SileroVAD_detect_silence(source_audio)
    clipping_blocks = comparator.SileroVAD_detect_silence(clipping_audio)

    # ブロック間の一致を確認
    matches = comparator.compare_audio_blocks(source_audio, clipping_audio, source_blocks, clipping_blocks)

    # 一致結果を出力
    for match in matches:
        print(f"Source: {match['source']}, Clip: {match['clip']}")
