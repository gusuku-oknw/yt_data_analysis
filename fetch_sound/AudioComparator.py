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

    def compare_audio_blocks(self, source_audio, clipping_audio, source_blocks, clipping_blocks, initial_threshold=100, threshold_increment=50):
        """
        切り抜き動画を基準に、元動画と一致する部分を探索。
        閾値を動的に増加させながら一致箇所を探します。
        一度一致したら次のブロックから探索を開始します。
        """
        matches = []
        current_threshold = initial_threshold
        source_index = 0  # 元動画の現在位置

        # キャッシュを利用してMFCC計算を最小化
        source_mfcc_cache = {}
        clip_mfcc_cache = {}

        for j, clip_block in tqdm(enumerate(clipping_blocks), total=len(clipping_blocks)):
            if j not in clip_mfcc_cache:
                clip_mfcc_cache[j] = self.extract_mfcc(
                    clipping_audio, clip_block["from"], clip_block["to"]
                )
            clip_mfcc = clip_mfcc_cache[j]

            match_found = False

            while not match_found:
                for i in range(source_index, len(source_blocks)):
                    if i not in source_mfcc_cache:
                        source_mfcc_cache[i] = self.extract_mfcc(
                            source_audio, source_blocks[i]["from"], source_blocks[i]["to"]
                        )
                    source_mfcc = source_mfcc_cache[i]

                    # fastdtwに1次元ベクトルを渡す
                    distance, _ = fastdtw(clip_mfcc, source_mfcc, dist=euclidean)

                    if distance < current_threshold:  # 閾値を使用
                        matches.append({
                            "clip_start": clip_block["from"],  # 切り抜き動画
                            "clip_end": clip_block["to"],
                            "source_start": source_blocks[i]["from"],  # 元動画
                            "source_end": source_blocks[i]["to"]
                        })
                        source_index = i + 1  # 次の比較を現在の位置から開始
                        match_found = True
                        break  # 一度一致したら次の切り抜きブロックへ

                if not match_found:
                    # 閾値を増加させる
                    current_threshold += threshold_increment

                    # 閾値が十分高い場合は一致なしと見なす
                    if current_threshold > 1000:  # 例として最大閾値を1000と設定
                        print(f"No match found for clip block {j} after raising threshold to {current_threshold}")
                        break

            # 次のブロックに進む前にキャッシュを削減
            if len(source_mfcc_cache) > 100:  # キャッシュサイズ制限
                source_mfcc_cache = {k: v for k, v in list(source_mfcc_cache.items())[-50:]}

        # 結果としてstartとendだけを出力
        return {
            "matches": [
                {
                    "clip": (match["clip_start"], match["clip_end"]),
                    "source": (match["source_start"], match["source_end"]),
                }
                for match in matches
            ],
            "final_threshold": current_threshold  # 最終的に使用された閾値を返す
        }


if __name__ == "__main__":
    # ファイルパスの設定
    source_audio = "../data/audio/source/bh4ObBry9q4.mp3"
    clipping_audio = "../data/audio/clipping/84iD1TEttV0.mp3"

    # サンプリングレートを取得してAudioComparatorのインスタンス作成
    comparator = AudioComparator(sampling_rate=torchaudio.info(source_audio).sample_rate)

    # VADを用いた音声区切りの取得
    source_blocks = comparator.SileroVAD_detect_silence(source_audio)
    clipping_blocks = comparator.SileroVAD_detect_silence(clipping_audio)

    # ブロック間の一致を確認
    result = comparator.compare_audio_blocks(source_audio, clipping_audio, source_blocks, clipping_blocks)

    # 一致結果を出力
    for match in result["matches"]:
        print(f"Source: {match['source']}, Clip: {match['clip']}")

    print(f"Final threshold used: {result['final_threshold']}")
