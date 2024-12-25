import os
import torch
import torchaudio
from transformers import Wav2Vec2Processor, Wav2Vec2Model
import logging

# ロギングの設定
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')


class Wav2VecAudioComparator:
    def __init__(self, model_name="facebook/wav2vec2-base-960h", sampling_rate=16000):
        self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        self.model = Wav2Vec2Model.from_pretrained(model_name).eval()
        self.sampling_rate = sampling_rate

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
        try:
            if not os.path.exists(audio_path):
                raise FileNotFoundError(f"音声ファイルが見つかりません: {audio_path}")

            # torchaudioでオーディオを読み込み、モノラルに変換
            waveform, sr = torchaudio.load(audio_path)
            logging.debug(f"Loaded waveform shape: {waveform.shape}, Sample rate: {sr}")
            if sr != self.sampling_rate:
                resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.sampling_rate)
                waveform = resampler(waveform)
                logging.debug(f"Resampled waveform shape: {waveform.shape}, New sample rate: {self.sampling_rate}")
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
                logging.debug(f"Converted to mono. Waveform shape: {waveform.shape}")

            audio_tensor = waveform.squeeze(0)

            speech_timestamps = self.get_speech_timestamps(
                audio_tensor, self.vad_model, threshold=threshold, sampling_rate=self.sampling_rate
            )

            logging.debug(f"検出された音声セグメント: {speech_timestamps}")

            if not speech_timestamps:
                logging.info(f"{audio_path}で音声が検出されませんでした。")
                return []

            # セグメントの時間単位を確認（サンプルインデックスであることを確認）
            valid_speech_timestamps = []
            for segment in speech_timestamps:
                start = segment['start']
                end = segment['end']
                if end < start:
                    logging.warning(f"セグメントのendがstartより小さいです: {segment}")
                    continue
                if end - start <= 0:
                    logging.warning(f"無効なセグメントが検出されました: {segment}")
                    continue
                valid_speech_timestamps.append(segment)

            return valid_speech_timestamps
        except Exception as e:
            logging.error(f"SileroVAD_detect_silence中でエラーが発生しました: {e}")
            return []

    def extract_features(self, audio_path, segment=None):
        try:
            waveform, sr = torchaudio.load(audio_path)
            logging.debug(f"Extract Features - Loaded waveform shape: {waveform.shape}, Sample rate: {sr}")
            if sr != self.sampling_rate:
                resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.sampling_rate)
                waveform = resampler(waveform)
                logging.debug(f"Extract Features - Resampled waveform shape: {waveform.shape}, New sample rate: {self.sampling_rate}")
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
                logging.debug(f"Extract Features - Converted to mono. Waveform shape: {waveform.shape}")

            if segment:
                start_sample = segment['start']
                end_sample = segment['end']
                logging.debug(f"セグメント開始サンプル: {start_sample}, 終了サンプル: {end_sample}")

                # サンプルインデックスが波形の範囲内に収まるように調整
                start_sample = max(0, min(start_sample, waveform.shape[1]))
                end_sample = max(0, min(end_sample, waveform.shape[1]))
                logging.debug(f"調整後のセグメント開始サンプル: {start_sample}, 終了サンプル: {end_sample}")

                waveform = waveform[:, start_sample:end_sample]
                logging.debug(f"Extract Features - Sliced waveform shape: {waveform.shape}")

            if waveform.shape[1] == 0:
                raise ValueError("セグメントの長さが0です。処理をスキップします。")

            input_values = self.processor(waveform.squeeze().numpy(), sampling_rate=self.sampling_rate,
                                          return_tensors="pt").input_values

            # 入力値の形状と内容をログに記録
            logging.debug(f"Input values shape: {input_values.shape}")
            logging.debug(f"Input values content: {input_values}")

            with torch.no_grad():
                features = self.model(input_values).last_hidden_state

            return features.squeeze(0)
        except Exception as e:
            logging.error(f"extract_features中でエラーが発生しました: {e}")
            raise

    def compare_audio_features(self, features1, features2, threshold=0.5):
        try:
            similarity = torch.nn.functional.cosine_similarity(features1.mean(dim=0), features2.mean(dim=0), dim=0)
            logging.debug(f"Cosine similarity: {similarity.item()}")
            return similarity > threshold
        except Exception as e:
            logging.error(f"compare_audio_features中でエラーが発生しました: {e}")
            return False

    def process_audio(self, source_audio, clipping_audio):
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
                try:
                    clip_features = self.extract_features(clipping_audio, segment=clip_segment)
                except ValueError as e:
                    logging.warning(f"クリップセグメントの処理をスキップします: {e}")
                    continue
                except Exception as e:
                    logging.error(f"クリップセグメントの特徴抽出中にエラーが発生しました: {e}")
                    continue

                match_found = False

                for source_segment in source_segments:
                    try:
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
                    except ValueError as e:
                        logging.warning(f"ソースセグメントの処理をスキップします: {e}")
                        continue
                    except Exception as e:
                        logging.error(f"ソースセグメントの特徴抽出中にエラーが発生しました: {e}")
                        continue

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
            # サンプルインデックスを秒に変換して表示
            clip_start_sec = block["clip_start"] / comparator.sampling_rate
            clip_end_sec = block["clip_end"] / comparator.sampling_rate
            source_start_sec = block["source_start"] / comparator.sampling_rate
            source_end_sec = block["source_end"] / comparator.sampling_rate
            print({
                "clip_start_sec": clip_start_sec,
                "clip_end_sec": clip_end_sec,
                "source_start_sec": source_start_sec,
                "source_end_sec": source_end_sec
            })
    else:
        print("マッチするブロックが見つかりませんでした。")
