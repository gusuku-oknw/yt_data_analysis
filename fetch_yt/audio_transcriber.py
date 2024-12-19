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
from tqdm import tqdm


class AudioTranscriber:
    def __init__(self):
        self.model_id = "kotoba-tech/kotoba-whisper-v1.0"

    @staticmethod
    def SileroVAD_detect_silence(audio_file, threshold=0.5, sampling_rate=16000):
        """
        Use Silero VAD to detect silence in an audio file.

        Parameters:
            audio_file (str): Path to the input audio file.
            threshold (float): Confidence threshold to distinguish speech from silence (0-1, default: 0.5).
            sampling_rate (int): Audio sampling rate (default: 16kHz).

        Returns:
            list: List of silence intervals [{'from': start_time, 'to': end_time, 'suffix': 'cut'}].
        """
        print(f"detect_silence: {audio_file}")
        try:
            # Load the Silero VAD model
            model, utils = torch.hub.load(
                repo_or_dir='snakers4/silero-vad',
                model='silero_vad',
                force_reload=True
            )

            # Unpack the utilities
            (get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks) = utils

            # Read the audio file
            if not os.path.exists(audio_file):
                raise FileNotFoundError(f"Audio file not found: {audio_file}")

            audio_tensor = read_audio(audio_file, sampling_rate=sampling_rate)
            print(f"Audio tensor size: {audio_tensor.shape}, dtype: {audio_tensor.dtype}")

            # Detect speech segments
            speech_timestamps = get_speech_timestamps(
                audio_tensor, model,
                threshold=threshold,
                sampling_rate=sampling_rate
            )
            print(f"Detected speech timestamps: {len(speech_timestamps)}")

            if not speech_timestamps:
                print(f"No speech detected in {audio_file}")
                return []

            # Calculate silence intervals
            silences = []
            last_end = 0
            audio_length = audio_tensor.shape[-1]
            for segment in speech_timestamps:
                if last_end < segment['start']:
                    silences.append({
                        "from": last_end / sampling_rate,  # Convert samples to seconds
                        "to": segment['start'] / sampling_rate,
                        "suffix": "cut"
                    })
                last_end = segment['end']

            # Add silence after the last speech segment
            if last_end < audio_length:
                print(f"Adding silence after last speech segment: {last_end} to {audio_length}")
                silences.append({
                    "from": last_end / sampling_rate,
                    "to": audio_length / sampling_rate,
                    "suffix": "cut"
                })

            return silences

        except Exception as e:
            print(f"An error occurred in SileroVAD_detect_silence: {e}")
            traceback.print_exc()
            return []

    @staticmethod
    def get_keep_blocks(silences, data_len, padding_time=0.2):
        """
        無音区間の外側を保持するブロックを計算。

        Parameters:
            silences (list): 無音区間のリスト（例: [{"from": 秒数, "to": 秒数}, ...]）。
            data_len (float): オーディオデータの長さ（秒単位）。
            samplerate (int): サンプルレート。
            padding_time (float): 保持する区間に加える余白時間（秒）。

        Returns:
            list: 保持する区間のリスト。
        """
        keep_blocks = []

        try:
            # 無音区間の外側を計算
            for i, block in enumerate(silences):
                if i == 0 and block["from"] > 0:
                    keep_blocks.append({"from": 0, "to": block["from"], "suffix": "keep"})
                if i > 0:
                    prev = silences[i - 1]
                    keep_blocks.append({"from": prev["to"], "to": block["from"], "suffix": "keep"})
                if i == len(silences) - 1 and block["to"] < data_len:
                    keep_blocks.append({"from": block["to"], "to": data_len, "suffix": "keep"})

            # 秒単位に変換し、余白を追加
            for block in keep_blocks:
                block["from"] = max(block["from"] - padding_time, 0)
                block["to"] = min(block["to"] + padding_time, data_len)

            return keep_blocks

        except Exception as e:
            print(f"Error in get_keep_blocks: {e}")
            print(f"Input silences: {silences}")
            print(f"Data length: {data_len}")
            print(traceback.format_exc())
            return []

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
        output_dir = os.path.join("../content", "audio_segments_{}".format(int(time.time())))

        # 出力ディレクトリが存在しない場合に作成
        os.makedirs(output_dir, exist_ok=True)

        segment_files = []

        # pydub AudioSegment を numpy.ndarray に変換
        data = np.array(audio.get_array_of_samples()) / (2 ** (audio.sample_width * 8 - 1))
        if audio.channels > 1:
            print("Stereo audio detected, converting to mono.")
            data = data.reshape((-1, audio.channels)).mean(axis=1)  # ステレオ -> モノラル

        for i, block in tqdm(enumerate(keep_blocks)):
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
                    # print(f"Saved segment {i}: {output_file}")
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
        # print("文字起こし結果:", result["text"])
        return result["text"]

    @staticmethod
    def extract_vocals(audio_file):
        """
        指定された音声ファイルからボーカルを抽出する。
        既にボーカルファイルが存在する場合はスキップ。
        """
        try:
            root_directory = os.path.dirname(audio_file)
            basename = os.path.splitext(os.path.basename(audio_file))[0]
            vocals_path = f"{root_directory}/htdemucs/{basename}/vocals.wav"

            # 既存ファイルのチェック
            if os.path.exists(vocals_path):
                print(f"ボーカルファイルが既に存在します: {vocals_path}")
                return vocals_path

            # ボーカル抽出を実行
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            print(f"Demucsを実行中 (デバイス: {device})...")

            command = ['demucs', '-d', device, '-o', root_directory, audio_file]
            subprocess.run(command, check=True)

            # ファイルの存在を再確認
            if os.path.exists(vocals_path):
                print(f"ボーカルファイルが生成されました: {vocals_path}")
                return vocals_path
            else:
                raise FileNotFoundError(f"ボーカルファイルが生成されませんでした: {vocals_path}")

        except subprocess.CalledProcessError as e:
            print(f"Demucs 実行中にエラーが発生しました: {e}")
        except FileNotFoundError as e:
            print(f"Demucs が見つかりません。インストールされていることを確認してください: {e}")
        except Exception as e:
            print(f"予期しないエラーが発生しました: {e}")

        return None

    def transcribe_audio_and_split_video(self, segment_files, keep_blocks):
        transcriptions = []

        num_segments = len(segment_files)

        for i in tqdm(range(num_segments)):
            block = keep_blocks[i]
            segment_start = block["from"]
            segment_end = block["to"]

            audio_file = segment_files[i]

            if not os.path.exists(audio_file):
                print(f"Audio segment file not found for block {i}: {audio_file}")
                continue

            try:
                # print(f"Processing {audio_file}")
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

    def transcribe_segment(self, audio_path, audio_extract=True):
        """
        音声ファイルを処理し、無音区間の外側を保持して文字起こしを行う。

        Parameters:
            audio_path (str): 入力音声ファイルのパス。

        Returns:
            list: 文字起こし結果。
        """
        print(f"Transcribing audio file: {audio_path}")
        # ボーカルのみを抽出
        if audio_extract:
            audio_path = self.extract_vocals(audio_path)

        # audio_path = 'data/sound/clipping_audio_wav/htdemucs/7-1fNxXj_xM/vocals.wav'
        # オーディオデータを読み込む
        print("Loading audio...")
        audio = AudioSegment.from_file(audio_path)

        silences = self.SileroVAD_detect_silence(audio_path)

        # サンプリングレートを取得
        samplerate = audio.frame_rate

        # 無音区間の外側を保持するブロックを計算
        keep_blocks = self.get_keep_blocks(
            silences=silences,
            data_len=len(audio.get_array_of_samples()),
            padding_time=0.2
        )

        print(f"Keep blocks: {len(keep_blocks)}")

        # 音声セグメントを保存
        segment_files = self.save_audio_segments(audio, keep_blocks, samplerate=samplerate)
        print(f"Segment files: {len(segment_files)}")

        # 各セグメントを文字起こし
        transcriptions = self.transcribe_audio_and_split_video(segment_files, keep_blocks)
        print(f"Transcriptions: {len(transcriptions)}")

        return transcriptions


if __name__ == "__main__":
    try:
        transcriber = AudioTranscriber()
        audio_path = '../data/sound/clipping_audio_wav/7-1fNxXj_xM.wav'
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        # 無音区間と文字起こしを行う
        silences = transcriber.transcribe_segment(audio_path)
        # print("Detected silences:")
        # for silence in silences:
        #     print(silence)
    except Exception as e:
        print(f"An error occurred: {e}")
        print(traceback.format_exc())
