import os
import torch
import torchaudio
from transformers import Wav2Vec2Processor, Wav2Vec2Model
import logging

# ロギングの設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class Wav2VecAudioComparator:
    def __init__(self, model_name="facebook/wav2vec2-base-960h", sampling_rate=16000):
        self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        self.model = Wav2Vec2Model.from_pretrained(model_name).eval()
        self.sampling_rate = sampling_rate

        # Silero VADの初期化
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

    def SileroVAD_detect_silence(self, audio_path, threshold=0.3):
        """
        Silero VADを使用して音声ファイルの無音部分を検出します。

        Parameters:
            audio_path (str): 音声ファイルのパス。
            threshold (float): 無音検出のしきい値。

        Returns:
            list: 音声ブロックのリスト。
        """
        try:
            if not os.path.exists(audio_path):
                raise FileNotFoundError(f"音声ファイルが見つかりません: {audio_path}")

            audio_tensor = self.read_audio(audio_path, sampling_rate=self.sampling_rate)

            speech_timestamps = self.get_speech_timestamps(
                audio_tensor, self.vad_model, threshold=threshold, sampling_rate=self.sampling_rate
            )

            if not speech_timestamps:
                logging.info(f"{audio_path}で音声が検出されませんでした。")
                return []

            return speech_timestamps
        except Exception as e:
            logging.error(f"SileroVAD_detect_silence中でエラーが発生しました: {e}")
            return []

    def extract_features(self, audio_path, segment=None):
        """
        Wav2Vec 2.0 を使用して音声特徴を抽出します。

        Parameters:
            audio_path (str): 音声ファイルのパス。
            segment (dict): セグメントの開始と終了時間。

        Returns:
            torch.Tensor: 音声特徴ベクトル。
        """
        waveform, sr = torchaudio.load(audio_path)
        if sr != self.sampling_rate:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.sampling_rate)
            waveform = resampler(waveform)

        if segment:
            start_sample = int(segment['start'] * self.sampling_rate)
            end_sample = int(segment['end'] * self.sampling_rate)
            waveform = waveform[:, start_sample:end_sample]

        input_values = self.processor(waveform.squeeze().numpy(), sampling_rate=self.sampling_rate, return_tensors="pt").input_values
        with torch.no_grad():
            features = self.model(input_values).last_hidden_state

        return features.squeeze(0)

    def compare_audio_features(self, features1, features2, threshold=0.5):
        """
        2つの音声特徴を比較します。

        Parameters:
            features1 (torch.Tensor): 音声1の特徴ベクトル。
            features2 (torch.Tensor): 音声2の特徴ベクトル。
            threshold (float): 類似性のしきい値。

        Returns:
            bool: 類似しているかどうか。
        """
        similarity = torch.nn.functional.cosine_similarity(features1.mean(dim=0), features2.mean(dim=0), dim=0)
        return similarity > threshold

    def process_audio(self, source_audio, clipping_audio):
        """
        ソース音声とクリッピング音声を処理し、マッチするブロックを取得します。

        Parameters:
            source_audio (str): ソース音声ファイルのパス。
            clipping_audio (str): クリッピング音声ファイルのパス。

        Returns:
            list: マッチしたブロックの詳細情報を含むリスト。
        """
        try:
            source_segments = self.SileroVAD_detect_silence(source_audio)
            clipping_segments = self.SileroVAD_detect_silence(clipping_audio)

            if not source_segments:
                logging.info("ソース音声で音声ブロックが検出されませんでした。")
                return []

            if not clipping_segments:
                logging.info("クリッピング音声で音声ブロックが検出されませんでした。")
                return []

            matches = []
            for clip_segment in clipping_segments:
                clip_features = self.extract_features(clipping_audio, segment=clip_segment)
                match_found = False

                for source_segment in source_segments:
                    source_features = self.extract_features(source_audio, segment=source_segment)
                    if self.compare_audio_features(clip_features, source_features, threshold=0.7):
                        matches.append({
                            "clip_start": clip_segment['start'],
                            "clip_end": clip_segment['end'],
                            "source_start": source_segment['start'],
                            "source_end": source_segment['end']
                        })
                        match_found = True
                        break

                if not match_found:
                    logging.info(f"クリップセグメント {clip_segment} のマッチが見つかりませんでした。")

            return matches
        except Exception as e:
            logging.error(f"process_audio中でエラーが発生しました: {e}", exc_info=True)
            return []

if __name__ == "__main__":
    source_audio = "../data/audio/source/bh4ObBry9q4.mp3"
    clipping_audio = "../data/audio/clipping/84iD1TEttV0.mp3"

    comparator = Wav2VecAudioComparator()
    result = comparator.process_audio(source_audio, clipping_audio)

    if result:
        print("マッチしたブロック:")
        for block in result:
            print(block)
    else:
        print("マッチするブロックが見つかりませんでした。")
