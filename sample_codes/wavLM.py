from transformers import Wav2Vec2FeatureExtractor, WavLMForXVector
import torch
import librosa
from itertools import product

# モデル名
model_name = "microsoft/wavlm-base-plus-sv"

# デバイス設定 (GPUが利用可能であれば使用)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# モデルと特徴抽出器のロード
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
model = WavLMForXVector.from_pretrained(model_name).to(device)

def split_audio(audio_path, segment_length, target_sr=16000):
    """
    Split audio into segments of the specified length.

    Args:
        audio_path (str): Path to the audio file.
        segment_length (int): Length of each segment in seconds.
        target_sr (int): Target sampling rate.

    Returns:
        list: List of audio segments.
    """
    waveform, sr = librosa.load(audio_path, sr=target_sr)
    total_length = len(waveform) / sr
    segments = []

    for start in range(0, int(total_length), segment_length):
        end = min(start + segment_length, total_length)
        start_sample = int(start * sr)
        end_sample = int(end * sr)
        segments.append(waveform[start_sample:end_sample])

    return segments

def calculate_similarity(segment1, segment2, sampling_rate=16000):
    """
    Calculate the similarity between two audio segments using WavLM.

    Args:
        segment1 (np.array): First audio segment.
        segment2 (np.array): Second audio segment.
        sampling_rate (int): Sampling rate of the audio segments.

    Returns:
        float: Cosine similarity between the two segments.
    """
    inputs = feature_extractor(
        [segment1, segment2],
        sampling_rate=sampling_rate,
        return_tensors="pt",
        padding=True,
    )

    # テンソルをGPUに転送
    inputs = {key: tensor.to(device) for key, tensor in inputs.items()}

    # 埋め込みを取得
    with torch.no_grad():
        embeddings = model(**inputs).embeddings

    # コサイン類似度の計算
    embeddings = torch.nn.functional.normalize(embeddings, dim=-1)
    cosine_sim = torch.nn.CosineSimilarity(dim=-1)

    return cosine_sim(embeddings[0], embeddings[1]).item()

def calculate_all_pair_similarities(audio_path1, audio_path2, segment_length, sampling_rate=16000):
    """
    Calculate the similarities between all segment pairs from two audio files.

    Args:
        audio_path1 (str): Path to the first audio file.
        audio_path2 (str): Path to the second audio file.
        segment_length (int): Length of each segment in seconds.
        sampling_rate (int): Sampling rate of the audio segments.

    Returns:
        list: List of similarities for all segment pairs.
    """
    segments1 = split_audio(audio_path1, segment_length, target_sr=sampling_rate)
    segments2 = split_audio(audio_path2, segment_length, target_sr=sampling_rate)

    similarities = []
    for i, j in product(range(len(segments1)), range(len(segments2))):
        similarity = calculate_similarity(segments1[i], segments2[j], sampling_rate=sampling_rate)
        similarities.append(((i, j), similarity))

    return similarities

if __name__ == "__main__":
    # 音声ファイルのパス
    source_audio_file = "../data/audio/source/Y1VQdn2pmgo.mp3"
    clip_audio_file = "../data/audio/clipping/qqxKJFDeNHg.mp3"

    # セグメント長さ (秒)
    segment_length = 5

    # 総当たりの類似度を計算
    all_similarities = calculate_all_pair_similarities(source_audio_file, clip_audio_file, segment_length)

    # 類似度を出力
    for (seg1_idx, seg2_idx), similarity in all_similarities:
        print(f"Segment ({seg1_idx}, {seg2_idx}): Similarity = {similarity:.4f}")
