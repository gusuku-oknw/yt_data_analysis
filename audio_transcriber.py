import os
import time
import librosa
import torch
from pydub import AudioSegment, silence  # silence モジュールをインポート
from transformers import pipeline
import soundfile as sf
import numpy as np
import traceback
import subprocess


class AudioTranscriber:
    def __init__(self):
        self.model_id = "kotoba-tech/kotoba-whisper-v1.0"

    # 無音部分の検出
    @staticmethod
    def detect_silence(audio_file, silence_thresh=-25, min_silence_len=50):
        """
        音声ファイルから無音部分を検出。

        Parameters:
            audio_file (str): 入力音声ファイルのパス。
            silence_thresh (int): 無音とみなす音量の閾値（デフォルト: -50dB）。
            min_silence_len (int): 無音とみなす最小の継続時間（デフォルト: 500ms）。

        Returns:
            list: 無音区間のリスト [{'from': 開始位置, 'to': 終了位置, 'suffix': 'cut'}]。
        """
        try:
            # 音声ファイルを読み込む
            audio = AudioSegment.from_file(audio_file)

            # 無音部分の開始・終了時間を検出
            silence_ranges = silence.detect_silence(
                audio,
                min_silence_len=min_silence_len,
                silence_thresh=silence_thresh
            )

            # 無音区間をリスト化
            silences = [
                {"from": start / 1000.0, "to": end / 1000.0, "suffix": "cut"}  # ミリ秒を秒に変換
                for start, end in silence_ranges
            ]

            return silences

        except Exception as e:
            print(f"エラーが発生しました: {e}")
            return []

    @staticmethod
    def get_keep_blocks(silences, data_len, samplerate, padding_time=0.2):
        """
        無音区間の外側を保持するブロックを計算。

        Parameters:
            silences (list): 無音区間のリスト（例: [[start, end], ...]）。
            data_len (int): オーディオデータの長さ（サンプル数）。
            samplerate (int): サンプルレート。
            padding_time (float): 保持する区間に加える余白時間（秒）。

        Returns:
            list: 保持する区間のリスト。
        """
        keep_blocks = []

        # 無音区間の外側を計算
        for i, block in enumerate(silences):
            if i == 0 and block[0] > 0:
                keep_blocks.append({"from": 0, "to": block[0], "suffix": "keep"})
            if i > 0:
                prev = silences[i - 1]
                keep_blocks.append({"from": prev[1], "to": block[0], "suffix": "keep"})
            if i == len(silences) - 1 and block[1] < data_len:
                keep_blocks.append({"from": block[1], "to": data_len, "suffix": "keep"})

        # 秒単位に変換し、余白を追加
        for block in keep_blocks:
            block["from"] = max(block["from"] / samplerate - padding_time, 0)
            block["to"] = min(block["to"] / samplerate + padding_time, data_len / samplerate)

        return keep_blocks

    @staticmethod
    def save_audio_segments(audio, keep_blocks, samplerate):
        """
        音声セグメントを指定のフォルダに保存します。

        Parameters:
            audio (AudioSegment): 入力音声データ（pydub AudioSegment オブジェクト）。
            keep_blocks (list): 保存する区間のリスト。
            samplerate (int): サンプルレート。

        Returns:
            list: 保存されたセグメントファイルの絶対パスのリスト。
        """
        # 修正された出力ディレクトリの設定
        output_dir = os.path.join("content", "audio_segments_{}".format(int(time.time())))

        # 出力ディレクトリが存在しない場合に作成
        os.makedirs(output_dir, exist_ok=True)

        segment_files = []

        # pydub AudioSegment を numpy.ndarray に変換
        data = np.array(audio.get_array_of_samples()) / (2 ** (audio.sample_width * 8 - 1))
        if audio.channels > 1:
            print("Stereo audio detected, converting to mono.")
            data = data.reshape((-1, audio.channels)).mean(axis=1)  # ステレオ -> モノラル

        for i, block in enumerate(keep_blocks):
            # サンプル単位の開始と終了位置を計算
            start_sample = int(block["from"] * samplerate)
            end_sample = int(block["to"] * samplerate)

            if start_sample < end_sample <= len(data):
                # セグメントを抽出
                segment_data = data[start_sample:end_sample]

                # 保存先のファイルパスを生成
                output_file = os.path.join(output_dir, f"segment_{i:02d}.wav")

                # ファイルを保存
                try:
                    sf.write(output_file, segment_data, samplerate)
                    segment_files.append(os.path.abspath(output_file))
                    print(f"Saved segment {i}: {output_file}")
                except Exception as e:
                    print(f"Error saving segment {i}: {e}")
            else:
                print(f"Skipped invalid segment {i}: start={start_sample}, end={end_sample}")

        return segment_files

    @staticmethod
    def kotoba_whisper(audio_file):
        """
        kotoba-Whisperを使用して音声を文字起こし。

        Parameters:
            audio_file (str): 入力音声ファイルのパス。

        Returns:
            str: 文字起こし結果。
        """
        model_id = "kotoba-tech/kotoba-whisper-v1.0"
        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        model_kwargs = {"attn_implementation": "sdpa"} if torch.cuda.is_available() else {}
        generate_kwargs = {
            "task": "transcribe",
            "return_timestamps": True,
            "language": "japanese"
        }

        pipe = pipeline(
            "automatic-speech-recognition",
            model=model_id,
            torch_dtype=torch_dtype,
            device=device,
            model_kwargs=model_kwargs
        )

        audio, sampling_rate = librosa.load(audio_file, sr=16000)
        audio_input = {"raw": audio, "sampling_rate": sampling_rate}
        result = pipe(audio_input, generate_kwargs=generate_kwargs)
        print("文字起こし結果:", result["text"])
        return result["text"]

    @staticmethod
    def extract_vocals(audio_file):
        # ディレクトリのルートを取得
        root_directory = os.path.dirname(audio_path)

        # ボーカルを抽出
        command = ["demucs", "-d", "cuda", "-o", root_directory, audio_file]
        subprocess.run(command, check=True)
        # ファイル名を取得
        basename = os.path.splitext(os.path.basename(audio_file))[0]

        return f"{root_directory}/htdemucs/{basename}/vocals.wav"

    def transcribe_audio_and_split_video(self, segment_files, keep_blocks):
        transcriptions = []

        num_segments = len(segment_files)

        for i in range(num_segments):
            block = keep_blocks[i]
            segment_start = block["from"]
            segment_end = block["to"]

            audio_file = segment_files[i]

            if not os.path.exists(audio_file):
                print(f"Audio segment file not found for block {i}: {audio_file}")
                continue

            try:
                print(f"Processing {audio_file}")
                transcription = self.kotoba_whisper(audio_file)
                transcriptions.append({
                    "text": transcription,
                    "start": segment_start,
                    "end": segment_end,
                    "audio_path": audio_file,
                })

            except Exception as e:
                error_message = f"Error during transcription for block {i}: {e}"
                print(error_message)
                print(traceback.format_exc())

                transcriptions.append({
                    "text": "",
                    "start": None,
                    "end": None,
                    "audio_path": audio_file,
                })

        return transcriptions

    def transcribe_segment(self, audio_path):
        """
        音声ファイルを処理し、無音区間の外側を保持して文字起こしを行う。

        Parameters:
            audio_path (str): 入力音声ファイルのパス。

        Returns:
            list: 文字起こし結果。
        """

        try:
            # ボーカルのみを抽出
            audio_path = self.extract_vocals(audio_path)

            # オーディオデータを読み込む
            audio = AudioSegment.from_file(audio_path)

            # ステレオデータの場合はモノラルに変換
            if audio.channels > 1:
                print("Stereo audio detected, converting to mono.")
                audio = audio.set_channels(1)

            # 無音区間を検出
            percentile = 99
            amplitudes = np.abs(np.array(audio.get_array_of_samples()))

            # 振幅のパーセンタイルをdBFSに変換
            silence_thresh = max(-40, 20 * np.log10(np.percentile(amplitudes, percentile)))  # -40dBFSを下限
            print(f"Calculated silence threshold: -{silence_thresh} dBFS")

            # 無音区間を検出
            silences = silence.detect_silence(
                audio,
                min_silence_len=1100,  # 無音と判定する最小持続時間 (ms)
                silence_thresh=-silence_thresh  # 動的に計算した閾値
            )

            print(f"Detected silences: {silences}")

            # サンプリングレートを取得
            samplerate = audio.frame_rate

            # 無音区間の外側を保持するブロックを計算
            keep_blocks = self.get_keep_blocks(
                silences=silences,
                data_len=len(audio.get_array_of_samples()),
                samplerate=samplerate,
                padding_time=0.2
            )

            print(f"Keep blocks: {keep_blocks}")

            # 音声セグメントを保存
            segment_files = self.save_audio_segments(audio, keep_blocks, samplerate=samplerate)
            print(f"Segment files: {segment_files}")

            # 各セグメントを文字起こし
            transcriptions = self.transcribe_audio_and_split_video(segment_files, keep_blocks)
            print(f"Transcriptions: {transcriptions}")

            return transcriptions

        except Exception as e:
            print(f"Error processing audio file {audio_path}: {e}")
            return []


if __name__ == "__main__":
    try:
        transcriber = AudioTranscriber()
        audio_path = './data/sound/clipping_audio_wav/7-1fNxXj_xM.wav'
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        # 無音区間と文字起こしを行う
        silences = transcriber.transcribe_segment(audio_path)
        print("Detected silences:")
        for silence in silences:
            print(silence)
    except Exception as e:
        print(f"An error occurred: {e}")
        print(traceback.format_exc())
