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
                device=str(self.device),  # "cuda" or "cpu"
                compute_type = "int8_float16",  # 例: メモリ軽減オプション
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
            (
                self.get_speech_timestamps,  # 音声あり区間(話者区間)を取得する関数
                self.save_audio,  # 音声ファイルとして保存する関数
                self.read_audio,  # 音声ファイルからテンソルを読む関数
                self.vad_collate,  # DataLoaderなどで使用されるコラテ関数
                self.collect_chunks  # 長音声を切り分けるための関数
            ) = self.utils
            logging.info("Silero VAD model loaded successfully.")
        except Exception as e:
            logging.error(f"Error loading Silero VAD: {e}")
            raise e

        # Janomeのトークナイザ（後段のテキスト比較で使用）
        self.janome_tokenizer = Tokenizer()

    def get_speech_segments(self, audio_file, threshold: float = 0.5):
        """
        Silero VAD を使って「音声あり区間」を取得し、秒単位のリストを返す関数。

        戻り値の例:
        [
            {"start_sec": 0.0,  "end_sec": 3.2 },
            {"start_sec": 4.5,  "end_sec": 7.1 },
            ...
        ]
        """
        if not os.path.exists(audio_file):
            raise FileNotFoundError(f"Audio file not found: {audio_file}")

        # 音声をテンソルとして読み込み (サンプリングレート指定)
        audio_tensor = self.read_audio(audio_file, sampling_rate=self.sampling_rate)

        # Silero VADで「音声がある区間(サンプル単位)」を検出
        speech_timestamps = self.get_speech_timestamps(
            audio_tensor,
            self.model_vad,
            threshold=threshold,
            sampling_rate=self.sampling_rate
        )

        # サンプル単位 -> 秒単位 に変換
        speech_segments = []
        for seg in speech_timestamps:
            start_sec = seg["start"] / self.sampling_rate
            end_sec = seg["end"] / self.sampling_rate
            speech_segments.append({
                "start_sec": start_sec,
                "end_sec": end_sec
            })

        return speech_segments

    def transcribe_with_vad(self, audio_file, threshold=0.5, language="ja", beam_size=5):
        """
        1) Silero VADで音声を区間に分割
        2) 各区間ごとにfaster-whisperで文字起こし
        3) faster-whisperが返すセグメントの相対時刻に「区間の開始秒」を加算し、絶対時刻に変換
        4) すべてまとめて返す
        """
        # まずはVADで音声あり区間を取得
        speech_segments = self.get_speech_segments(audio_file, threshold=threshold)
        if not speech_segments:
            logging.warning("音声があるセグメントが検出されませんでした。")
            return []

        # librosa で「float32 numpy配列」として全体音声を読み込み
        audio, sr = librosa.load(audio_file, sr=self.sampling_rate)
        audio = audio.astype(np.float32)

        all_transcribed_segments = []

        # 各区間ごとに小分けした音声を作り、whisperで推論
        for seg in tqdm(speech_segments, desc="Transcribing each VAD segment"):
            seg_start_sec = seg["start_sec"]
            seg_end_sec = seg["end_sec"]

            # numpy配列におけるインデックス
            start_idx = int(seg_start_sec * sr)
            end_idx = int(seg_end_sec * sr)
            segment_audio = audio[start_idx:end_idx]

            # faster-whisper でこの小区間を文字起こし (seg.start, seg.end はブロック内での相対秒)
            segments, info = self.whisper_pipe.transcribe(
                segment_audio,
                language=language,
                beam_size=beam_size
            )

            # 各segmentについて、相対時刻に seg_start_sec を足して絶対時刻に変換
            for whisper_seg in segments:
                absolute_start = seg_start_sec + whisper_seg.start
                absolute_end = seg_start_sec + whisper_seg.end
                text = whisper_seg.text

                all_transcribed_segments.append({
                    "text": text,
                    "start": absolute_start,
                    "end": absolute_end
                })

        return all_transcribed_segments

    # ==========================================
    # 以下、テキスト比較に関わる関数 (お好みで)
    # ==========================================

    def preprocess_text(self, text):
        """テキストの前処理（スペース削除、記号削除など）"""
        text = re.sub(r'\s+', '', text)  # 連続する空白の削除
        text = re.sub(r'[^\w\s]', '', text)  # 記号削除
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

# -------------------------------------------------
# (C) メイン実行例
# -------------------------------------------------
if __name__ == "__main__":
    source_audio = "../data/audio/source/pnHdRQbR2zs.mp3"
    clipping_audio = "../data/audio/clipping/-bRcKCM5_3E.mp3"

    comparator = WhisperComparison(sampling_rate=16000)

    # 1. ソース音声をセグメント化＆文字起こし
    source_segments = comparator.transcribe_with_vad(
        audio_file=source_audio,
        threshold=0.5,  # VADの閾値
        language="ja",
        beam_size=5
    )
    print("\n=== Source Segments ===")
    for i, seg in enumerate(source_segments):
        print(f"({i+1}) [{seg['start']:.2f}s - {seg['end']:.2f}s] {seg['text']}")

    # 2. 切り抜き音声をセグメント化＆文字起こし
    clipping_segments = comparator.transcribe_with_vad(
        audio_file=clipping_audio,
        threshold=0.5,
        language="ja",
        beam_size=5
    )
    print("\n=== Clipping Segments ===")
    for i, seg in enumerate(clipping_segments):
        print(f"({i+1}) [{seg['start']:.2f}s - {seg['end']:.2f}s] {seg['text']}")

    # 3. 両者のテキストセグメントを比較 (切り抜き → ソース)
    #    fast_method="sequence" と slow_method="tfidf" を使い、閾値0.8でマッチングを試みる
    matches, unmatched = comparator.compare_segments(
        clipping_segments,
        source_segments,
        initial_threshold=0.8,
        fast_method="sequence",
        slow_method="tfidf"
    )

    # 4. 結果表示
    print("\n=== マッチ結果 ===")
    if matches:
        for match_list in matches:
            for sub in match_list:
                print(f"[Clip] start={sub['clip_start']:.2f}s end={sub['clip_end']:.2f}s, text=\"{sub['clip_text']}\"")
                print(f" -> [Source] start={sub['source_start']:.2f}s end={sub['source_end']:.2f}s, text=\"{sub['source_text']}\"")
                print(f" -> similarity={sub['similarity']:.3f}\n")
    else:
        print("マッチなし")
