import torch
import torchaudio
import librosa
import numpy as np
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import torchaudio.transforms as transforms

# デバイス設定
device = "cuda" if torch.cuda.is_available() else "cpu"

# モデルとトークナイザーのロード
model_name = "../models/clap-htsat-fused"
model = AutoModel.from_pretrained(model_name).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name)

print("モデルがロードされました:", model)

def load_audio(file_path, target_sr=48000):
    """
    音声ファイルを読み込む
    """
    audio, sr = librosa.load(file_path, sr=target_sr)
    return audio, sr

def preprocess_audio(audio, sr, target_sr=48000):
    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)
        audio = resampler(torch.tensor(audio))
    else:
        audio = torch.tensor(audio)
    print(f"Preprocessed audio shape: {audio.shape}")
    try:
        mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=target_sr,
            n_fft=1024,
            hop_length=256,
            n_mels=128
        )
        mel_audio = mel_spectrogram(audio.unsqueeze(0))
        print(f"Mel spectrogram shape: {mel_audio.shape}")
        return mel_audio.unsqueeze(0)
    except Exception as e:
        print(f"Error in MelSpectrogram: {e}")
        return None



def get_embeddings(segments, model, device):
    embeddings = []
    for i, segment in enumerate(segments):
        if len(segment) == 0:
            print(f"Segment {i} is empty, skipping.")
            continue
        try:
            audio_tensor = preprocess_audio(segment, sr=48000).to(device)
            print(f"Segment {i}: Processed audio tensor shape: {audio_tensor.shape}")
            with torch.no_grad():
                outputs = model(audio_tensor)
                if outputs is None:
                    print(f"Segment {i}: Model returned None.")
                    continue
                embedding = outputs.last_hidden_state.cpu().numpy()
                embeddings.append(embedding)
        except Exception as e:
            print(f"Segment {i}: Error processing segment: {e}")
    print(f"Generated {len(embeddings)} embeddings.")
    return embeddings



def compare_blocks(source_embeddings, clip_embeddings):
    """
    ソースとクリップの埋め込み間のコサイン類似度を計算
    """
    # Check for empty embeddings
    if not source_embeddings or not clip_embeddings:
        raise ValueError("One or both embedding lists are empty. Cannot compute similarity.")

    # Concatenate embeddings and compute cosine similarity
    try:
        similarity_matrix = cosine_similarity(
            np.vstack(source_embeddings), np.vstack(clip_embeddings)
        )
        return similarity_matrix
    except Exception as e:
        print(f"Error computing similarity matrix: {e}")
        return None

def segment_audio(audio, sr, segment_duration=10.0):
    """
    音声データを指定された長さのセグメントに分割する
    """
    total_duration = len(audio) / sr
    segments = []
    for start in np.arange(0, total_duration, segment_duration):
        end = min(start + segment_duration, total_duration)
        start_sample = int(start * sr)
        end_sample = int(end * sr)
        segments.append(audio[start_sample:end_sample])
    return segments

# 音声ファイルのパス
source_audio_file = "../data/audio/source/Y1VQdn2pmgo.mp3"
clip_audio_file = "../data/audio/clipping/qqxKJFDeNHg.mp3"

# 音声データの読み込み
source_audio, source_sr = load_audio(source_audio_file)
clip_audio, clip_sr = load_audio(clip_audio_file)

# 音声データをセグメント化
source_segments = segment_audio(source_audio, source_sr, segment_duration=10.0)
clip_segments = segment_audio(clip_audio, clip_sr, segment_duration=10.0)

# 各セグメントの埋め込みを取得
source_embeddings = get_embeddings(source_segments, model, device)
clip_embeddings = get_embeddings(clip_segments, model, device)

# セグメント間の類似度を計算
similarity_matrix = compare_blocks(source_embeddings, clip_embeddings)

# 結果を表示
similarity_df = pd.DataFrame(
    similarity_matrix,
    index=[f"Source_{i}" for i in range(len(source_embeddings))],
    columns=[f"Clip_{i}" for i in range(len(clip_embeddings))]
)
print(similarity_df)
