import os
import torch
import torchaudio
import subprocess
from transformers import Wav2Vec2Processor, Wav2Vec2Model
import logging
from typing import List, Dict

# ロギング設定
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

class Wav2VecAudioComparator:
    def __init__(self, model_name="facebook/wav2vec2-base-960h", sampling_rate=16000, vad_model_name='silero_vad'):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        self.model = Wav2Vec2Model.from_pretrained(model_name).to(self.device).eval()
        self.sampling_rate = sampling_rate

        # Silero VADの初期化
        try:
            self.vad_model, self.vad_utils = torch.hub.load(
                repo_or_dir='snakers4/silero-vad',
                model='silero_vad',
                force_reload=False
            )
            self.get_speech_timestamps, self.save_audio, self.read_audio, _, _ = self.vad_utils
            self.vad_model.to(self.device)
        except Exception as e:
            logging.error(f"Silero VADモデルの読み込み中にエラーが発生しました: {e}")
            raise

    def extract_vocals(self, audio_file: str) -> str:
        """
        Demucsを使用して音声ファイルからボーカルを抽出します。
        既にボーカルファイルが存在する場合はスキップします。
        """
        try:
            root_directory = os.path.dirname(audio_file)
            basename = os.path.splitext(os.path.basename(audio_file))[0]
            vocals_path = os.path.join(root_directory, "htdemucs", basename, "vocals.wav")

            # 既存ファイルのチェック
            if os.path.exists(vocals_path):
                logging.info(f"ボーカルファイルが既に存在します: {vocals_path}")
                return vocals_path

            # ボーカル抽出を実行
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            logging.debug(f"Demucsを実行中 (デバイス: {device})...")

            command = ['demucs', '-d', device, '-o', root_directory, audio_file]
            subprocess.run(command, check=True)

            # ファイルの存在を再確認
            if os.path.exists(vocals_path):
                logging.info(f"ボーカルファイルが生成されました: {vocals_path}")
                return vocals_path
            else:
                raise FileNotFoundError(f"ボーカルファイルが生成されませんでした: {vocals_path}")

        except subprocess.CalledProcessError as e:
            logging.error(f"Demucs 実行中にエラーが発生しました: {e}")
        except FileNotFoundError as e:
            logging.error(f"Demucs が見つかりません。インストールされていることを確認してください: {e}")
        except Exception as e:
            logging.error(f"予期しないエラーが発生しました: {e}")

        return ""

    def get_speech_timestamps_custom(self, audio_tensor: torch.Tensor, threshold: float = 0.3) -> List[Dict]:
        """
        Silero VADを使用して音声セグメントのタイムスタンプを取得します。
        """
        try:
            logging.debug("get_speech_timestamps_custom - VADを使用して音声セグメントを検出中...")
            speech_timestamps = self.get_speech_timestamps(audio_tensor, self.vad_model, threshold=threshold, sampling_rate=self.sampling_rate)
            logging.debug(f"get_speech_timestamps_custom - 検出された音声セグメント数: {len(speech_timestamps)}")
            return speech_timestamps
        except Exception as e:
            logging.error(f"get_speech_timestamps_customでエラーが発生しました: {e}")
            return []

    def SileroVAD_detect_silence(self, audio_file: str, threshold: float = 0.3) -> List[Dict]:
        """
        Silero VADを使用して音声ファイルの無音部分を検出します。
        音声セグメントのリストを返します。
        """
        try:
            if not os.path.exists(audio_file):
                raise FileNotFoundError(f"音声ファイルが見つかりません: {audio_file}")

            audio_tensor = self.read_audio(audio_file, sampling_rate=self.sampling_rate).to(self.device)

            speech_timestamps = self.get_speech_timestamps_custom(audio_tensor, threshold=threshold)

            if not speech_timestamps:
                logging.info(f"{audio_file}で音声が検出されませんでした。")
                return []

            silences = []
            last_end = 0
            audio_length = audio_tensor.shape[-1] / self.sampling_rate  # 秒単位

            for segment in speech_timestamps:
                start_time = segment['start'] / self.sampling_rate
                end_time = segment['end'] / self.sampling_rate
                if last_end < start_time:
                    silences.append({
                        "from": last_end,
                        "to": start_time,
                        "suffix": "cut"
                    })
                last_end = end_time

            if last_end < audio_length:
                silences.append({
                    "from": last_end,
                    "to": audio_length,
                    "suffix": "cut"
                })

            return silences
        except Exception as e:
            logging.error(f"SileroVAD_detect_silence中でエラーが発生しました: {e}")
            return []

    def extract_features(self, audio_path: str, segments: List[Dict] = None) -> torch.Tensor:
        """
        音声ファイルまたは音声セグメントから特徴量を抽出します。
        セグメントが指定されている場合、各セグメントごとに特徴量を抽出し、平均を取ります。
        """
        try:
            waveform, sr = torchaudio.load(audio_path)
            logging.debug(f"extract_features - Loaded waveform shape: {waveform.shape}, Sample rate: {sr}")
            if sr != self.sampling_rate:
                resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.sampling_rate)
                waveform = resampler(waveform)
                logging.debug(f"extract_features - Resampled waveform shape: {waveform.shape}, New sample rate: {self.sampling_rate}")
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
                logging.debug(f"extract_features - Converted to mono. Waveform shape: {waveform.shape}")

            features_list = []

            if segments:
                logging.debug(f"extract_features - 処理するセグメント数: {len(segments)}")
                for i, segment in enumerate(segments):
                    start_sample = int(segment['from'] * self.sampling_rate)
                    end_sample = int(segment['to'] * self.sampling_rate) if segment['to'] else waveform.shape[-1]
                    logging.debug(f"extract_features - セグメント {i+1}: {start_sample} から {end_sample} サンプルまで")

                    segment_waveform = waveform[:, start_sample:end_sample]
                    if segment_waveform.shape[1] == 0:
                        logging.warning(f"セグメント {i+1} の長さが0です。スキップします。")
                        continue

                    input_values = self.processor(segment_waveform.squeeze().cpu().numpy(), sampling_rate=self.sampling_rate,
                                                return_tensors="pt").input_values.to(self.device)

                    logging.debug(f"extract_features - セグメント {i+1} の入力値形状: {input_values.shape}")

                    with torch.no_grad():
                        features = self.model(input_values).last_hidden_state

                    features_list.append(features.squeeze(0))
            else:
                logging.debug("extract_features - セグメントが指定されていないため、全体を処理します。")
                input_values = self.processor(waveform.squeeze().cpu().numpy(), sampling_rate=self.sampling_rate,
                                            return_tensors="pt").input_values.to(self.device)

                logging.debug(f"extract_features - 入力値形状: {input_values.shape}")

                with torch.no_grad():
                    features = self.model(input_values).last_hidden_state

                features_list.append(features.squeeze(0))

            if not features_list:
                raise ValueError("全てのセグメントの長さが0です。処理をスキップします。")

            # 全セグメントの特徴量を平均して1つのベクトルにする
            aggregated_features = torch.stack(features_list).mean(dim=0)

            logging.debug(f"extract_features - 集約された特徴量の形状: {aggregated_features.shape}")

            return aggregated_features

        except Exception as e:
            logging.error(f"extract_features中でエラーが発生しました: {e}")
            raise

    def compare_audio_features(self, features1: torch.Tensor, features2: torch.Tensor, threshold: float = 0.5) -> bool:
        """
        二つの特徴量ベクトル間のコサイン類似度を計算し、閾値を基に類似性を評価します。
        """
        try:
            similarity = torch.nn.functional.cosine_similarity(features1.mean(dim=0), features2.mean(dim=0), dim=0)
            logging.debug(f"compare_audio_features - 類似度: {similarity.item()}")
            return similarity.item() > threshold
        except Exception as e:
            logging.error(f"compare_audio_features中でエラーが発生しました: {e}")
            return False

    def process_audio(self, source_audio: str, clipping_audio: str, vad_threshold: float = 0.3, similarity_threshold: float = 0.7) -> Dict:
        """
        ソース音声とクリッピング音声を比較し、各ブロックの類似度を計算し、
        最も類似度が高いブロックをリスト化して返します。
        """
        try:
            # クリッピング音声からボーカルを抽出
            clipping_vocals = self.extract_vocals(clipping_audio)
            if not clipping_vocals:
                logging.error("切り抜き動画のボーカル抽出に失敗しました。")
                return {"vocals_match": False, "most_similar_segments": []}

            # ソース音声の無音部分を検出
            silences_source = self.SileroVAD_detect_silence(source_audio, threshold=vad_threshold)

            # ソース音声を無音部分で分割して音声セグメントを取得
            if silences_source:
                logging.debug("process_audio - ソース音声を無音部分で分割します。")
                speech_segments = []
                last_end = 0
                for silence in silences_source:
                    from_time = silence["from"]
                    to_time = silence["to"]
                    # 音声セグメントは無音部分の前
                    if last_end < from_time:
                        speech_segments.append({
                            "from": last_end,
                            "to": from_time,
                            "suffix": "speech"
                        })
                    last_end = to_time
                # 最後の無音部分後の音声セグメント
                audio_length = self.read_audio(source_audio, sampling_rate=self.sampling_rate).shape[-1] / self.sampling_rate
                if last_end < audio_length:
                    speech_segments.append({
                        "from": last_end,
                        "to": audio_length,
                        "suffix": "speech"
                    })
            else:
                # 無音部分が検出されなかった場合、全体を一つのセグメントとして処理
                logging.debug("process_audio - 無音部分が検出されなかったため、全体を一つのセグメントとして処理します。")
                speech_segments = [{"from": 0, "to": None, "suffix": "speech"}]

            # ソース音声の音声セグメントから特徴量を抽出
            segment_similarity = []
            if speech_segments:
                for i, segment in enumerate(speech_segments):
                    features_source = self.extract_features(source_audio, segments=[segment])
                    features_clipping = self.extract_features(clipping_vocals)
                    similarity = self.compare_audio_features(features_source, features_clipping)
                    segment_similarity.append({
                        "segment_index": i,
                        "from": segment["from"],
                        "to": segment["to"],
                        "similarity": similarity
                    })
            else:
                # セグメントがない場合、全体を処理
                features_source = self.extract_features(source_audio)
                features_clipping = self.extract_features(clipping_vocals)
                similarity = self.compare_audio_features(features_source, features_clipping)
                segment_similarity.append({
                    "segment_index": 0,
                    "from": 0,
                    "to": None,
                    "similarity": similarity
                })

            # 類似度が最も高いセグメントを取得
            most_similar_segments = sorted(segment_similarity, key=lambda x: x["similarity"], reverse=True)

            result = {
                "vocals_match": any(seg["similarity"] > similarity_threshold for seg in segment_similarity),
                "most_similar_segments": most_similar_segments
            }

            if result["vocals_match"]:
                logging.info("切り抜き動画のボーカルが元動画と一致するセグメントが見つかりました。")
            else:
                logging.info("一致するボーカルセグメントが見つかりませんでした。")

            return result

        except Exception as e:
            logging.error(f"process_audio中でエラーが発生しました: {e}", exc_info=True)
            return {"vocals_match": False, "most_similar_segments": []}

if __name__ == "__main__":
    source_audio = "../data/audio/source/Y1VQdn2pmgo.mp3"
    clipping_audio = "../data/audio/clipping/qqxKJFDeNHg.mp3"

    comparator = Wav2VecAudioComparator()
    result = comparator.process_audio(source_audio, clipping_audio)

    if result.get("vocals_match"):
        print("切り抜き動画のボーカルが元動画と一致しました。")
    else:
        print("一致するボーカルが見つかりませんでした。")

    # 将来的に非ボーカルの比較を追加する場合は、以下を有効にしてください。
    # if result.get("no_vocals_match"):
    #     print("切り抜き動画の非ボーカルが元動画と一致しました。")
    # else:
    #     print("一致する非ボーカルが見つかりませんでした。")
