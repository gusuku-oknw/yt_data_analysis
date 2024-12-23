import os
import torch
import torchaudio
from torchaudio.transforms import MFCC
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from tqdm import tqdm
import numpy as np
from transformers import pipeline
import librosa

class AudioComparator:
    def __init__(self, sampling_rate=16000, n_mfcc=13):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.sampling_rate = sampling_rate
        self.mfcc_transform = MFCC(
            sample_rate=sampling_rate,
            n_mfcc=n_mfcc
        ).to(self.device)

        # Silero VAD model loading once
        self.vad_model, self.vad_utils = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad',
            force_reload=False
        )
        self.get_speech_timestamps, self.save_audio, self.read_audio, _, _ = self.vad_utils

        # Kotoba-Whisper model pipeline
        self.kotoba_whisper_model_id = "kotoba-tech/kotoba-whisper-v1.0"
        self.kotoba_whisper_torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        self.kotoba_whisper_device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.kotoba_whisper_model_kwargs = {"attn_implementation": "sdpa"} if torch.cuda.is_available() else {}
        self.kotoba_whisper_pipe = pipeline(
            "automatic-speech-recognition",
            model=self.kotoba_whisper_model_id,
            torch_dtype=self.kotoba_whisper_torch_dtype,
            device=self.kotoba_whisper_device,
            model_kwargs=self.kotoba_whisper_model_kwargs
        )

    def SileroVAD_detect_silence(self, audio_file, threshold=0.5):
        """Detect silence in an audio file using Silero VAD."""
        try:
            if not os.path.exists(audio_file):
                raise FileNotFoundError(f"Audio file not found: {audio_file}")

            audio_tensor = self.read_audio(audio_file, sampling_rate=self.sampling_rate)

            speech_timestamps = self.get_speech_timestamps(
                audio_tensor, self.vad_model, threshold=threshold, sampling_rate=self.sampling_rate
            )

            if not speech_timestamps:
                print(f"No speech detected in {audio_file}")
                return []

            silences = []
            last_end = 0
            audio_length = audio_tensor.shape[-1]
            for segment in speech_timestamps:
                if last_end < segment['start']:
                    silences.append({
                        "from": last_end / self.sampling_rate,
                        "to": segment['start'] / self.sampling_rate,
                        "suffix": "cut"
                    })
                last_end = segment['end']

            if last_end < audio_length:
                silences.append({
                    "from": last_end / self.sampling_rate,
                    "to": audio_length / self.sampling_rate,
                    "suffix": "cut"
                })

            return silences
        except Exception as e:
            print(f"An error occurred in SileroVAD_detect_silence: {e}")
            return []

    def compute_full_mfcc(self, audio_path):
        """Compute MFCC for the entire audio file."""
        waveform, file_sr = torchaudio.load(audio_path)
        if file_sr != self.sampling_rate:
            resampler = torchaudio.transforms.Resample(orig_freq=file_sr, new_freq=self.sampling_rate).to(self.device)
            waveform = waveform.to(self.device)  # デバイスを統一
            waveform = resampler(waveform)
        else:
            waveform = waveform.to(self.device)  # デバイスを統一

        # MFCC計算
        mfcc = self.mfcc_transform(waveform)
        return mfcc.squeeze(0)

    def extract_mfcc_block(self, full_mfcc, start, end):
        """Extract MFCC for a specific time block."""
        hop_length = 512  # Default hop length
        start_frame = int(start * self.sampling_rate / hop_length)
        end_frame = int(end * self.sampling_rate / hop_length)
        block_mfcc = full_mfcc[:, start_frame:end_frame]
        return block_mfcc.mean(dim=-1).cpu().numpy()

    def kotoba_whisper(self, audio_file, max_segment_duration=30.0):
        """
        kotoba-Whisperを使用して音声を文字起こし。

        Parameters:
            audio_file (str): 入力音声ファイルのパス。
            max_segment_duration (float): 最大セグメント長（秒）。

        Returns:
            str: 文字起こし結果。
        """
        generate_kwargs = {
            "return_timestamps": True,
            "language": "japanese"
        }

        audio, sampling_rate = librosa.load(audio_file, sr=self.sampling_rate)
        total_duration = len(audio) / self.sampling_rate
        segments = []

        # 音声をセグメントに分割
        for start in np.arange(0, total_duration, max_segment_duration):
            end = min(start + max_segment_duration, total_duration)
            start_sample = int(start * self.sampling_rate)
            end_sample = int(end * self.sampling_rate)
            segments.append(audio[start_sample:end_sample])

        # 各セグメントを処理
        transcriptions = []
        for segment in segments:
            audio_input = {"raw": segment, "sampling_rate": self.sampling_rate}
            result = self.kotoba_whisper_pipe(audio_input, generate_kwargs=generate_kwargs)
            transcriptions.append(result["text"])

        # セグメントの文字起こしを結合
        return " ".join(transcriptions)

    def transcribe_blocks(self, audio_file, blocks):
        """
        Transcribe audio blocks using kotoba-whisper.

        Parameters:
            audio_file (str): Path to the audio file.
            blocks (list): List of blocks with "from" and "to" times.

        Returns:
            list: List of transcribed text for each block.
        """
        transcriptions = []
        audio, sr = librosa.load(audio_file, sr=self.sampling_rate)
        for block in tqdm(blocks, desc="Transcribing Blocks"):
            start_sample = int(block["from"] * sr)
            end_sample = int(block["to"] * sr)
            block_audio = audio[start_sample:end_sample]

            audio_input = {"raw": block_audio, "sampling_rate": sr}
            result = self.kotoba_whisper_pipe(audio_input,
                                              generate_kwargs={"return_timestamps": True, "language": "japanese"})
            transcriptions.append(result["text"])

        return transcriptions

    def calculate_audio_statistics(self, audio_file, blocks):
        """
        Calculate audio statistics (average loudness and variance) for the entire audio and each block.

        Parameters:
            audio_file (str): Path to the audio file.
            blocks (list): List of blocks with "from" and "to" times.

        Returns:
            dict: Average loudness of the entire audio and variance for each block.
        """
        audio, sr = librosa.load(audio_file, sr=self.sampling_rate)
        overall_mean = np.mean(np.abs(audio))

        block_statistics = []
        for block in blocks:
            start_sample = int(block["from"] * sr)
            end_sample = int(block["to"] * sr)
            block_audio = audio[start_sample:end_sample]
            block_variance = np.var(block_audio)
            block_statistics.append({
                "from": block["from"],
                "to": block["to"],
                "variance": block_variance
            })

        return {
            "overall_mean": overall_mean,
            "block_statistics": block_statistics
        }

    def compare_audio_blocks(self, source_audio, clipping_audio, source_blocks, clipping_blocks, initial_threshold=100, threshold_increment=50):
        """Compare audio blocks between source and clipping audio."""
        source_full_mfcc = self.compute_full_mfcc(source_audio)
        clipping_full_mfcc = self.compute_full_mfcc(clipping_audio)

        matches = []
        current_threshold = initial_threshold
        source_index = 0

        for j in tqdm(range(len(clipping_blocks)), desc="Processing Blocks"):
            clip_block = clipping_blocks[j]
            clip_mfcc = self.extract_mfcc_block(clipping_full_mfcc, clip_block["from"], clip_block["to"])
            match_found = False

            while not match_found:
                for i in range(source_index, len(source_blocks)):
                    source_block = source_blocks[i]
                    source_mfcc = self.extract_mfcc_block(source_full_mfcc, source_block["from"], source_block["to"])

                    distance = fastdtw(clip_mfcc, source_mfcc, dist=euclidean)[0]

                    if distance < current_threshold:
                        matches.append({
                            "clip_start": clip_block["from"],
                            "clip_end": clip_block["to"],
                            "source_start": source_block["from"],
                            "source_end": source_block["to"]
                        })
                        source_index = i + 1
                        match_found = True
                        break

                if not match_found:
                    current_threshold += threshold_increment
                    if current_threshold > 1000:
                        print(f"No match found for clip block {j} after raising threshold to {current_threshold}")
                        break

        return {
            "matches": [
                {
                    "clip": (match["clip_start"], match["clip_end"]),
                    "source": (match["source_start"], match["source_end"]),
                }
                for match in matches
            ],
            "final_threshold": current_threshold
        }

if __name__ == "__main__":
    source_audio = "../data/audio/source/bh4ObBry9q4.mp3"
    clipping_audio = "../data/audio/clipping/84iD1TEttV0.mp3"

    comparator = AudioComparator()

    source_blocks = comparator.SileroVAD_detect_silence(source_audio)
    clipping_blocks = comparator.SileroVAD_detect_silence(clipping_audio)

    transcriptions = comparator.transcribe_blocks(source_audio, source_blocks)
    for i, transcription in enumerate(transcriptions):
        print(f"Block {i + 1}: {transcription}")

    audio_statistics = comparator.calculate_audio_statistics(source_audio, source_blocks)
    print("Overall Mean Loudness:", audio_statistics["overall_mean"])
    for block_stat in audio_statistics["block_statistics"]:
        print(f"Block {block_stat['from']}-{block_stat['to']} Variance: {block_stat['variance']}")

    result = comparator.compare_audio_blocks(source_audio, clipping_audio, source_blocks, clipping_blocks)

    for match in result["matches"]:
        print(f"Source: {match['source']}, Clip: {match['clip']}")

    print(f"Final threshold used: {result['final_threshold']}")
