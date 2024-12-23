import os
import torch
import torchaudio
from transformers import WavLMConfig, WavLMModel
from scipy.spatial.distance import cosine
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor


class AudioComparator:
    def __init__(self, sampling_rate=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # サンプリングレートが指定されていない場合、デフォルト値を設定
        self.sampling_rate = sampling_rate if sampling_rate else 16000

        # WavLM設定の初期化
        configuration = WavLMConfig()
        self.model = WavLMModel(configuration).to(self.device)

    def get_sampling_rate(self, audio_path):
        """
        音声ファイルのサンプリングレートを取得
        """
        info = torchaudio.info(audio_path)
        return info.sample_rate

    def SileroVAD_detect_silence(self, audio_file, threshold=0.5):
        """
        Silero VADを使って音声の区切り（静寂部分）を検出。
        """
        print(f"detect_silence: {audio_file}")
        try:
            # Silero VADモデルをロード
            model, utils = torch.hub.load(
                repo_or_dir="snakers4/silero-vad",
                model="silero_vad",
                force_reload=True,
            )
            (get_speech_timestamps, save_audio, read_audio, _, _) = utils

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
                if last_end < segment["start"]:
                    silences.append(
                        {
                            "from": last_end / self.sampling_rate,
                            "to": segment["start"] / self.sampling_rate,
                            "suffix": "cut",
                        }
                    )
                last_end = segment["end"]

            if last_end < audio_length:
                silences.append(
                    {
                        "from": last_end / self.sampling_rate,
                        "to": audio_length / self.sampling_rate,
                        "suffix": "cut",
                    }
                )

            return silences
        except Exception as e:
            print(f"An error occurred in SileroVAD_detect_silence: {e}")
            return []

    def extract_embedding(self, audio_path, start, end):
        """
        指定された音声区間のWavLM埋め込みを抽出。
        """
        # フレームオフセットとフレーム数を計算
        frame_offset = int(start * self.sampling_rate)
        num_frames = int((end - start) * self.sampling_rate)

        # 音声ファイルをロード
        waveform, sr = torchaudio.load(audio_path, frame_offset=frame_offset, num_frames=num_frames)

        # サンプリングレートが異なる場合はリサンプリング
        if sr != self.sampling_rate:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.sampling_rate)
            waveform = resampler(waveform)

        # モデルに入力するためにテンソルを準備
        inputs = waveform.mean(dim=0).unsqueeze(0).to(self.device)  # チャンネル次元を統合しバッチ次元を追加

        with torch.no_grad():
            embeddings = self.model(inputs).last_hidden_state
            mean_embedding = embeddings.mean(dim=1).squeeze().cpu().numpy()

        return mean_embedding

    def extract_embedding_parallel(self, audio_path, blocks):
        """
        並列処理で複数の音声区間の埋め込みを抽出。
        """
        with ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(self.extract_embedding, audio_path, block["from"], block["to"])
                for block in blocks
            ]
            embeddings = [future.result() for future in futures]
        return embeddings

    def compare_audio_blocks(self, source_audio, clipping_audio, source_blocks, clipping_blocks):
        """
        切り抜き動画を基準に、元動画と一致する部分を探索（コサイン類似度を使用）。
        並列処理を使用して埋め込みを取得。
        """
        matches = []
        source_embeddings = self.extract_embedding_parallel(source_audio, source_blocks)
        clip_embeddings = self.extract_embedding_parallel(clipping_audio, clipping_blocks)

        source_index = 0  # 元動画の現在位置
        for j, clip_embedding in tqdm(enumerate(clip_embeddings), total=len(clip_embeddings)):
            for i in range(source_index, len(source_embeddings)):
                source_embedding = source_embeddings[i]
                similarity = 1 - cosine(clip_embedding, source_embedding)
                if similarity > 0.85:  # 類似度の閾値を調整
                    matches.append({
                        "clip_start": clipping_blocks[j]["from"],  # 切り抜き動画
                        "clip_end": clipping_blocks[j]["to"],
                        "source_start": source_blocks[i]["from"],  # 元動画
                        "source_end": source_blocks[i]["to"]
                    })
                    source_index = i + 1
                    break

        return [
            {
                "clip": (match["clip_start"], match["clip_end"]),
                "source": (match["source_start"], match["source_end"]),
            }
            for match in matches
        ]


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
    matches = comparator.compare_audio_blocks(source_audio, clipping_audio, source_blocks, clipping_blocks)

    # 一致結果を出力
    for match in matches:
        print(f"Source: {match['source']}, Clip: {match['clip']}")
