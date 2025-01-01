import os
import re
import logging
import torch
import torchaudio
import librosa
import numpy as np
from tqdm import tqdm
from faster_whisper import WhisperModel

from janome.tokenizer import Tokenizer
from difflib import SequenceMatcher
from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.feature_extraction.text import TfidfVectorizer


# ロギングの設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class WhisperComparison:
    def __init__(self, sampling_rate: int = 16000):
        # CUDAが使える場合はcuda、そうでない場合はcpu
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.sampling_rate = sampling_rate

        # faster-whisperモデルの読み込み
        try:
            self.whisper_pipe = WhisperModel(
                model_size_or_path="deepdml/faster-whisper-large-v3-turbo-ct2",
                device=str(self.device)  # "cuda" or "cpu"
            )
            logging.info("faster-whisper model loaded successfully.")
        except Exception as e:
            logging.error(f"Error loading Whisper Model: {e}")
            raise e

        # Silero VADモデルの読み込み
        try:
            self.model_vad, self.utils = torch.hub.load(
                repo_or_dir='snakers4/silero-vad',
                model='silero_vad',
                force_reload=True
            )
            (self.get_speech_timestamps, _, self.read_audio, _, _) = self.utils
            logging.info("Silero VAD model loaded successfully.")
        except Exception as e:
            logging.error(f"Error loading Silero VAD: {e}")
            raise e

        # Janomeのトークナイザ
        self.janome_tokenizer = Tokenizer()

    def detect_silence(self, audio_input, threshold: float = 0.4):
        """
        Silero VAD を用いて無音区間を検出
        """
        # audio_inputがファイルパスの場合
        if isinstance(audio_input, str):
            if not os.path.exists(audio_input):
                raise FileNotFoundError(f"Audio file not found: {audio_input}")
            audio_tensor = self.read_audio(audio_input, sampling_rate=self.sampling_rate)
        else:
            # すでにテンソルやnumpy配列の場合
            audio_tensor = audio_input

        if len(audio_tensor) == 0:
            logging.warning("Audio tensor is empty. No silence detected.")
            return []

        # 無音区間を検出
        speech_timestamps = self.get_speech_timestamps(
            audio_tensor,
            self.model_vad,
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

        # 音声末尾以降の無音区間があれば追加
        if last_end < len(audio_tensor):
            silences.append({
                "from": last_end / self.sampling_rate,
                "to": len(audio_tensor) / self.sampling_rate
            })

        return silences

    def transcribe_blocks(self, audio_file, blocks, language: str = "ja", beam_size: int = 5):
        """
        faster-whisperを使用して音声ブロックを文字起こしします。
        blocks は [{"from": start_sec, "to": end_sec}, ...] のリスト。
        """
        transcriptions = []
        try:
            # librosaで読み込み
            audio, sr = librosa.load(audio_file, sr=self.sampling_rate)

            for block in tqdm(blocks, desc="Transcribing Blocks"):
                start_sample = int(block["from"] * sr)
                end_sample = int(block["to"] * sr)

                # ブロックごとの音声切り出し
                block_audio = audio[start_sample:end_sample]

                # float32に変換 (faster-whisperが推奨)
                block_audio = block_audio.astype(np.float32)

                # transcribe
                segments, info = self.whisper_pipe.transcribe(
                    block_audio,
                    language=language,
                    beam_size=beam_size
                )

                # セグメントをまとめてテキスト化
                block_text = "".join([seg.text for seg in segments])

                transcriptions.append({
                    "text": block_text,
                    "start": block["from"],
                    "end": block["to"]
                })

        except Exception as e:
            logging.error(f"transcribe_blocks中でエラーが発生しました: {e}", exc_info=True)

        return transcriptions

    def transcribe_audio(self, audio_file, language="ja", beam_size=5):
        """
        音声ファイル全体を文字起こししてテキストとして返します。
        """
        try:
            audio, sr = librosa.load(audio_file, sr=self.sampling_rate)
            # float32に変換 (faster-whisperが推奨)
            audio = audio.astype(np.float32)

            segments, info = self.whisper_pipe.transcribe(
                audio,
                language=language,
                beam_size=beam_size
            )

            # セグメントをまとめてテキスト化
            text = "".join([seg.text for seg in segments])
            return text.strip()
        except Exception as e:
            logging.error(f"Error transcribing audio: {e}", exc_info=True)
            return ""

    # =========================
    # 以下、テキスト比較関連のメソッド
    # =========================

    def preprocess_text(self, text):
        """テキストの前処理（スペース削除、記号削除など）"""
        text = re.sub(r'\s+', '', text)       # 連続する空白の削除
        text = re.sub(r'[^\w\s]', '', text)   # 記号削除
        return text

    def tokenize_japanese(self, text):
        """Janomeで日本語をトークナイズ"""
        tokens = [token.surface for token in self.janome_tokenizer.tokenize(text)]
        return tokens

    def calculate_similarity(self, text1, text2, method="sequence"):
        """与えられた2つのテキスト間の類似度を計算"""
        text1 = self.preprocess_text(text1)
        text2 = self.preprocess_text(text2)

        # 空文字の場合は類似度0
        if not text1.strip() or not text2.strip():
            return 0.0

        if method == "sequence":
            return SequenceMatcher(None, text1, text2).ratio()

        elif method == "jaccard":
            set1 = set(self.tokenize_japanese(text1))
            set2 = set(self.tokenize_japanese(text2))
            intersection = len(set1 & set2)
            union = len(set1 | set2)
            return intersection / union if union != 0 else 0.0

        elif method == "tfidf":
            vectorizer = TfidfVectorizer(tokenizer=self.tokenize_japanese)
            try:
                tfidf_matrix = vectorizer.fit_transform([text1, text2]).toarray()
                numerator = np.dot(tfidf_matrix[0], tfidf_matrix[1])
                denominator = np.linalg.norm(tfidf_matrix[0]) * np.linalg.norm(tfidf_matrix[1])
                return numerator / denominator if denominator != 0 else 0.0
            except ValueError:
                return 0.0

        else:
            raise ValueError("Unsupported similarity method")

    def find_best_match(self, segment, source_segments, used_segments,
                        threshold=0.8, method="tfidf"):
        """
        1つの切り抜きセグメント `segment` に対して、
        最も類似度が高いかつ threshold 以上を満たす元動画セグメントを探す。
        """
        best_match = None
        max_similarity = 0

        for src in source_segments:
            # すでに使用したセグメントはスキップ
            if src["start"] in used_segments:
                continue

            similarity = self.calculate_similarity(segment["text"], src["text"], method=method)
            if similarity > max_similarity and similarity >= threshold:
                best_match = src
                max_similarity = similarity

        return best_match

    def compare_segments(self, clipping_segments, source_segments,
                         initial_threshold=0.8, fast_method="sequence",
                         slow_method="tfidf"):
        """
        切り抜きセグメント(clipping_segments)と元動画セグメント(source_segments)を比較し、
        まず fast_method でマッチングを試み、マッチしなかったものは slow_method で再マッチング。
        """
        matches = []
        unmatched = []
        used_segments = set()

        def process_clip(clip, method):
            try:
                threshold = initial_threshold
                found_match = False
                clip_matches = []

                # もしテキストが長い場合は50文字ごとに分割（必要に応じて調整）
                long_segments = [clip["text"]]
                if len(clip["text"]) > 50:
                    long_segments = [
                        clip["text"][i:i + 50]
                        for i in range(0, len(clip["text"]), 50)
                    ]
                # 空白だけのセグメントは除外
                long_segments = [seg for seg in long_segments if seg.strip()]

                # 分割したテキストブロックごとにマッチングを試す
                for segment_text in long_segments:
                    local_threshold = threshold
                    while local_threshold > 0.1:
                        candidate = {
                            "text": segment_text,
                            "start": clip["start"],
                            "end": clip["end"]
                        }
                        best = self.find_best_match(candidate, source_segments,
                                                    used_segments,
                                                    threshold=local_threshold,
                                                    method=method)
                        if best:
                            similarity_score = self.calculate_similarity(segment_text, best["text"], method=method)
                            clip_matches.append({
                                "clip_text": segment_text,
                                "clip_start": clip["start"],
                                "clip_end": clip["end"],
                                "source_text": best["text"],
                                "source_start": best["start"],
                                "source_end": best["end"],
                                "similarity": similarity_score,
                            })
                            used_segments.add(best["start"])
                            found_match = True
                            break
                        local_threshold -= 0.05

                if not found_match:
                    # このクリップセグメントはマッチングできなかった
                    return {
                        "clip_text": clip["text"],
                        "clip_start": clip["start"],
                        "clip_end": clip["end"],
                        "matched": False
                    }

                return {"matches": clip_matches, "matched": True}

            except Exception as e:
                logging.error(f"Error processing clip: {clip.get('text', 'Unknown')}, {e}")
                return {"error": str(e), "matched": False}

        # Step 1: fast_method でのマッチング
        logging.info("Fast methodでのマッチングを実行中...")
        with ThreadPoolExecutor() as executor:
            futures = {
                executor.submit(process_clip, clip, fast_method): clip
                for clip in clipping_segments
            }

            for future in tqdm(as_completed(futures),
                               total=len(clipping_segments),
                               desc="Comparing segments (fast_method)"):
                result = future.result()
                if result.get("matched"):
                    matches.append(result["matches"])
                else:
                    unmatched.append(result)

        # Step 2: slow_method での再マッチング (unmatched のみ)
        if unmatched:
            logging.info("Unmatchedに対してslow methodでのマッチングを実行中...")
            still_unmatched = []
            with ThreadPoolExecutor() as executor:
                futures = {
                    executor.submit(process_clip, u, slow_method): u
                    for u in unmatched
                }
                for future in tqdm(as_completed(futures),
                                   total=len(unmatched),
                                   desc="Comparing segments (slow_method)"):
                    result = future.result()
                    if result.get("matched"):
                        matches.append(result["matches"])
                    else:
                        still_unmatched.append(result)
            unmatched = still_unmatched

        return matches, unmatched

    def process_audio(self, source_audio_file, clipping_audio_file,
                      fast_method="sequence", slow_method="tfidf", threshold=0.8):
        """
        2つの音声ファイルを文字起こしし、それぞれを単一セグメント化したもの同士で比較。
        （必要に応じて、無音区間検出や複数ブロックへの分割を適用してください）
        """
        # 1. ソース音声を文字起こし
        source_text = self.transcribe_audio(source_audio_file)
        # 2. 切り抜き音声を文字起こし
        clipping_text = self.transcribe_audio(clipping_audio_file)

        # ここではシンプルに「音声全体を1つのセグメント」にしています。
        source_segments = [{
            "text": source_text,
            "start": 0,
            "end": 0
        }]
        clipping_segments = [{
            "text": clipping_text,
            "start": 0,
            "end": 0
        }]

        # 3. 上記2つのセグメントを比較
        matches, unmatched = self.compare_segments(
            clipping_segments,
            source_segments,
            initial_threshold=threshold,
            fast_method=fast_method,
            slow_method=slow_method
        )

        # 必要ならここで出力を整形する
        return matches, unmatched


if __name__ == "__main__":
    source_audio = "../data/audio/source/pnHdRQbR2zs.mp3"
    clipping_audio = "../data/audio/clipping/-bRcKCM5_3E.mp3"

    try:
        # torchaudioを使ってサンプリングレートを取得
        sampling_rate = torchaudio.info(clipping_audio).sample_rate
    except Exception as e:
        logging.error(f"Error reading audio info: {e}")
        sampling_rate = 16000  # デフォルトのサンプリングレートを設定

    comparator = WhisperComparison(sampling_rate=sampling_rate)

    # process_audio は (matches, unmatched) のタプルを返す
    matches, unmatched = comparator.process_audio(source_audio, clipping_audio)

    print("\n=== マッチ結果 ===")
    if matches:
        for match_list in matches:
            for sub in match_list:
                print(f"[Clip] start={sub['clip_start']} end={sub['clip_end']}, text={sub['clip_text']}")
                print(f" -> [Source] start={sub['source_start']} end={sub['source_end']}, text={sub['source_text']}")
                print(f" -> similarity={sub['similarity']:.3f}\n")
    else:
        print("マッチなし")

    print("\n=== マッチしなかったセグメント ===")
    if unmatched:
        for um in unmatched:
            print(f"[Unmatched Clip] start={um['clip_start']} end={um['clip_end']}, text={um['clip_text']}")
    else:
        print("なし")
