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
                # force_reload=True
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

    def transcribe_with_vad(self, audio_file, threshold=0.5, language="ja", beam_size=5, padding=0.2):
        """
        1) Silero VADで音声を区間に分割
        2) 各区間ごとにfaster-whisperで文字起こし
        3) faster-whisperが返すセグメントの相対時刻に「区間の開始秒」を加算し、絶対時刻に変換
        4) すべてまとめて返す

        Parameters:
            audio_file (str): 入力音声ファイル
            threshold (float): VADの閾値
            language (str): 文字起こし言語
            beam_size (int): Whisperのビームサイズ
            padding (float): 各セグメントに追加する前後の無音区間（秒）

        Returns:
            list: 文字起こし結果を含むセグメントリスト
        """
        # まずはVADで音声あり区間を取得
        speech_segments = self.get_speech_segments(audio_file, threshold=threshold)
        if not speech_segments:
            logging.warning("音声があるセグメントが検出されませんでした。")
            return []

        # librosa で「float32 numpy配列」として全体音声を読み込み
        audio, sr = librosa.load(audio_file, sr=self.sampling_rate)
        audio = audio.astype(np.float32)
        total_duration = len(audio) / sr

        all_transcribed_segments = []

        # 各区間ごとに小分けした音声を作り、whisperで推論
        for seg in tqdm(speech_segments, desc="Transcribing each VAD segment"):
            seg_start_sec = max(0, seg["start_sec"] - padding)  # 前にpaddingを追加
            seg_end_sec = min(total_duration, seg["end_sec"] + padding)  # 後ろにpaddingを追加

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
    # 以下、テキスト比較に関わる関数
    # ==========================================

    def preprocess_text(self, text):
        """テキストの前処理（スペース削除、記号削除など）"""
        text = re.sub(r'\s+', '', text)     # 連続する空白の削除
        text = re.sub(r'[^\w\s]', '', text) # 記号削除
        return text

    def tokenize_japanese(self, text):
        """Janomeで日本語をトークナイズ"""
        tokens = [token.surface for token in self.janome_tokenizer.tokenize(text)]
        return tokens

    def calculate_similarity(self, text1, text2, method="sequence"):
        """
        与えられた2つのテキスト間の類似度を計算。
        method に "another_method" が指定された場合は、
        例として単純な単語数の差をベースにしたダミー類似度など、
        別のロジックが動くようにしています。
        """
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

        elif method == "another_method":
            # 例: 別のモデルやロジックを使うという想定のダミー実装
            # 実際には他の音声認識モデル／NLPモデルの呼び出しなどに置き換えてください。
            len_diff = abs(len(text1) - len(text2))
            # 長さが似ていれば類似度が高いというテキトーな例
            sim = 1.0 - (len_diff / max(len(text1), len(text2)))
            return max(0.0, min(sim, 1.0))

        else:
            raise ValueError(f"Unsupported similarity method: {method}")

    def find_best_match(self, segment, source_segments, used_segments,
                        threshold=0.8, method="sequence"):
        """
        1つの切り抜きセグメント `segment` に対して、
        最も類似度が高いかつ threshold 以上を満たす元動画セグメントを探す。
        既にused_segmentsに登録されたstart秒のソースは再利用しない。
        """
        best_match = None
        max_similarity = 0.0

        for src in source_segments:
            if src["start"] in used_segments:
                continue

            similarity = self.calculate_similarity(segment["text"], src["text"], method=method)
            if similarity > max_similarity and similarity >= threshold:
                best_match = src
                max_similarity = similarity

        return best_match, max_similarity

    def compare_segments(self, clipping_segments, source_segments,
                         initial_threshold=0.8,
                         fast_method="sequence",
                         slow_method="tfidf"):
        """
        切り抜きセグメント(clipping_segments)と元動画セグメント(source_segments)を比較し、
        まず fast_method でマッチングを試み、マッチしなかったものは slow_method で再マッチング。

        今回の改良点:
        1) fast / slow いずれかでしきい値を下回った場合、"another_method" を呼ぶ
        2) 一致したソースは source_segments から除外 (再利用しない)
        3) 文字数の多い順にクリップを比較
        4) Unmatched は「一致しなかったソース側のセグメント」を返す
        """

        # 文字数の多いクリップから処理する
        # reverse=True で降順ソート
        clipping_segments = sorted(clipping_segments, key=lambda x: len(x["text"]), reverse=True)

        matches = []
        # 今回は「マッチしなかったクリップ」を保持せず、Source_segment だけを最後に返す
        # つまり、"unmatched_clips" は作らない

        used_segments = set()  # ここにマッチ済みソースの start 秒を記録して再利用を防ぐ

        def process_clip(clip, method, threshold):
            """
            1つのclipに対して指定methodでマッチングを試みる。
            - threshold を下回る場合は "another_method" で再試行する。
            - 最初にマッチしたら終了。
            """
            # テキストが長い場合の分割（例: 50文字区切り）
            if len(clip["text"]) > 50:
                long_segments = [
                    clip["text"][i:i + 50]
                    for i in range(0, len(clip["text"]), 50)
                ]
            else:
                long_segments = [clip["text"]]

            # 空白だけの区切りはスキップ
            long_segments = [seg for seg in long_segments if seg.strip()]

            matched_infos = []
            found_match_any = False

            for seg_text in long_segments:
                local_threshold = threshold
                # しきい値を徐々に下げながら探索
                while local_threshold > 0.1:
                    candidate = {
                        "text": seg_text,
                        "start": clip["start"],
                        "end": clip["end"]
                    }
                    best, sim_score = self.find_best_match(candidate,
                                                           source_segments,
                                                           used_segments,
                                                           local_threshold,
                                                           method)
                    if best:
                        matched_infos.append({
                            "clip_text": seg_text,
                            "clip_start": clip["start"],
                            "clip_end": clip["end"],
                            "source_text": best["text"],
                            "source_start": best["start"],
                            "source_end": best["end"],
                            "similarity": sim_score,
                        })
                        # マッチした source は使い終わったので再利用しない
                        used_segments.add(best["start"])
                        if best in source_segments:
                            source_segments.remove(best)
                        found_match_any = True
                        break
                    else:
                        local_threshold -= 0.05

                # fast / slow でも見つからなかったら、"another_method" で再試行
                if not found_match_any:
                    candidate = {
                        "text": seg_text,
                        "start": clip["start"],
                        "end": clip["end"]
                    }
                    best, sim_score = self.find_best_match(candidate,
                                                           source_segments,
                                                           used_segments,
                                                           0.0,  # 最後なのでほぼ下限値
                                                           "another_method")
                    if best and sim_score >= 0.1:
                        matched_infos.append({
                            "clip_text": seg_text,
                            "clip_start": clip["start"],
                            "clip_end": clip["end"],
                            "source_text": best["text"],
                            "source_start": best["start"],
                            "source_end": best["end"],
                            "similarity": sim_score,
                        })
                        used_segments.add(best["start"])
                        if best in source_segments:
                            source_segments.remove(best)
                        found_match_any = True

            if found_match_any:
                return {
                    "matches": matched_infos,
                    "matched": True
                }
            else:
                return {
                    "matched": False
                }

        # ------------------------------
        #   Step 1: fast_method でのマッチング
        # ------------------------------
        logging.info("Fast methodでのマッチングを実行中...")

        # シンプルにシングルスレッドで実行
        unmatched_clips = []  # fast でマッチしなかったクリップ (仮で保持)
        for clip in clipping_segments:
            result = process_clip(clip, fast_method, initial_threshold)
            if result["matched"]:
                # matchesリストに追加
                matches.extend(result["matches"])
            else:
                unmatched_clips.append(clip)

        # ------------------------------
        #   Step 2: slow_method で再マッチング
        # ------------------------------
        if unmatched_clips:
            logging.info("Unmatchedに対してslow methodでのマッチングを実行中...")
            still_unmatched_clips = []
            for clip in unmatched_clips:
                result = process_clip(clip, slow_method, initial_threshold)
                if result["matched"]:
                    matches.extend(result["matches"])
                else:
                    still_unmatched_clips.append(clip)

            # ここで still_unmatched_clips は最終的にクリップがマッチしなかったもの
            # しかし、今回はクリップ未マッチの情報は返さず、無視する

        # --------------------------------------
        #   最終的に余っている source_segments が「未使用ソース」
        # --------------------------------------
        # matches で取り除かなかったもの = 一度も使われなかった source_segments
        unmatched_source = []
        for seg in source_segments:
            unmatched_source.append({
                "source_text": seg["text"],
                "source_start": seg["start"],
                "source_end": seg["end"]
            })

        return matches, unmatched_source

# -------------------------------------------------
# (C) メイン実行例
# -------------------------------------------------
if __name__ == "__main__":
    source_audio = "../data/audio/source/-1FjQf5ipUA.mp3"
    clipping_audio = "../data/audio/clipping/htdemucs/FIoqPoclE6M/vocals.wav"

    comparator = WhisperComparison(sampling_rate=16000)

    # 1. ソース音声をセグメント化＆文字起こし
    source_segments = comparator.transcribe_with_vad(
        audio_file=source_audio,
        threshold=0.35,  # VADの閾値
        language="ja",
        beam_size=5
    )
    print("\n=== Source Segments ===")
    for i, seg in enumerate(source_segments):
        print(f"({i+1}) [{seg['start']:.2f}s - {seg['end']:.2f}s] {seg['text']}")

    # 2. 切り抜き音声をセグメント化＆文字起こし
    clipping_segments = comparator.transcribe_with_vad(
        audio_file=clipping_audio,
        threshold=0.35,
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
