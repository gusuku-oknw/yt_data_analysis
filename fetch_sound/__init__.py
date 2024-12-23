import os
import librosa
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
import torch

class AudioComparator:
    @staticmethod
    def SileroVAD_detect_silence(audio_file, threshold=0.5, sampling_rate=16000):
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

            audio_tensor = read_audio(audio_file, sampling_rate=sampling_rate)

            # 音声セグメントを検出
            speech_timestamps = get_speech_timestamps(
                audio_tensor, model,
                threshold=threshold,
                sampling_rate=sampling_rate
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
                        "from": last_end / sampling_rate,  # サンプルを秒に変換
                        "to": segment['start'] / sampling_rate,
                        "suffix": "cut"
                    })
                last_end = segment['end']

            if last_end < audio_length:
                silences.append({
                    "from": last_end / sampling_rate,
                    "to": audio_length / sampling_rate,
                    "suffix": "cut"
                })

            return silences

        except Exception as e:
            print(f"An error occurred in SileroVAD_detect_silence: {e}")
            return []

    @staticmethod
    def extract_mfcc(audio_path, start, end, sr=16000, n_mfcc=13):
        """
        指定された区間のMFCCを抽出。
        """
        y, _ = librosa.load(audio_path, sr=sr, offset=start, duration=end - start)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        return mfcc.T

    @staticmethod
    def compare_audio_blocks(source_audio, clipping_audio, source_blocks, clipping_blocks):
        """
        区切られたブロック同士で一致を確認し、startとendのみを出力。
        """
        matches = []
        for i, source_block in enumerate(source_blocks):
            source_mfcc = AudioComparator.extract_mfcc(
                source_audio, source_block["from"], source_block["to"]
            )
            for j, clip_block in enumerate(clipping_blocks):
                clip_mfcc = AudioComparator.extract_mfcc(
                    clipping_audio, clip_block["from"], clip_block["to"]
                )
                distance, _ = fastdtw(source_mfcc, clip_mfcc, dist=euclidean)
                if distance < 100:  # 閾値を調整
                    matches.append({
                        "source_start": source_block["from"],
                        "source_end": source_block["to"],
                        "clip_start": clip_block["from"],
                        "clip_end": clip_block["to"]
                    })

        # 結果としてstartとendだけを出力
        return [
            {
                "source": (match["source_start"], match["source_end"]),
                "clip": (match["clip_start"], match["clip_end"]),
            }
            for match in matches
        ]


# ファイルパスの設定
source_audio = "../data/audio/source/bh4ObBry9q4.mp3"
clipping_audio = "../data/audio/clipping/84iD1TEttV0.mp3"

# VADを用いた音声区切りの取得
source_blocks = AudioComparator.SileroVAD_detect_silence(source_audio)
clipping_blocks = AudioComparator.SileroVAD_detect_silence(clipping_audio)

# ブロック間の一致を確認
matches = AudioComparator.compare_audio_blocks(source_audio, clipping_audio, source_blocks, clipping_blocks)

# 一致結果を出力
for match in matches:
    print(f"Source: {match['source']}, Clip: {match['clip']}")
