from transformers import Wav2Vec2FeatureExtractor, WavLMForXVector
import torch
import librosa
import os
import logging
from itertools import product

# モデル名
model_name = "microsoft/wavlm-base-plus-sv"

# デバイス設定 (GPUが利用可能であれば使用)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# モデルと特徴抽出器のロード
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
model = WavLMForXVector.from_pretrained(model_name).to(device)

# Silero VAD モデルの読み込み
def load_silero_vad():
    try:
        vad_model, vad_utils = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad',
            force_reload=False
        )
        get_speech_timestamps, save_audio, read_audio, _, _ = vad_utils
        return vad_model, get_speech_timestamps, read_audio
    except Exception as e:
        logging.error(f"Silero VADモデルの読み込み中にエラーが発生しました: {e}")
        raise

vad_model, get_speech_timestamps, read_audio = load_silero_vad()

def detect_silence(audio_path, sampling_rate=16000, threshold=0.5):
    """
    Silero VADを使用して無音部分を検出します。

    Args:
        audio_path (str): Path to the audio file.
        sampling_rate (int): Sampling rate of the audio.
        threshold (float): VAD threshold.

    Returns:
        list: List of silent segments with start and end times.
    """
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"音声ファイルが見つかりません: {audio_path}")

    audio_tensor = read_audio(audio_path, sampling_rate=sampling_rate)
    speech_timestamps = get_speech_timestamps(audio_tensor, vad_model, threshold=threshold, sampling_rate=sampling_rate)

    silences = []
    last_end = 0
    audio_length = audio_tensor.shape[-1]

    for segment in speech_timestamps:
        if last_end < segment['start']:
            silences.append({
                "from": last_end / sampling_rate,
                "to": segment['start'] / sampling_rate,
                "suffix": "cut"
            })
        last_end = segment['end']

    if last_end < audio_length:
        silences.append({
            "from": last_end / sampling_rate,
            "to": audio_length / sampling_rate,
            "suffix": "cut"
        })

    return silences

def split_audio_with_silence(audio_path, sampling_rate=16000, threshold=0.5, min_segment_length=0.1):
    """
    音声を無音部分で分割し、最小長さを満たすセグメントのみを保持します。

    Args:
        audio_path (str): Path to the audio file.
        sampling_rate (int): Sampling rate of the audio.
        threshold (float): VAD threshold.
        min_segment_length (float): Minimum segment length in seconds.

    Returns:
        list: List of audio segments.
    """
    silences = detect_silence(audio_path, sampling_rate=sampling_rate, threshold=threshold)
    audio_tensor = read_audio(audio_path, sampling_rate=sampling_rate)
    segments = []

    last_end = 0
    for silence in silences:
        start_sample = int(last_end * sampling_rate)
        end_sample = int(silence["from"] * sampling_rate)
        segment = audio_tensor[start_sample:end_sample]
        # 最小長さを確認
        if len(segment) / sampling_rate >= min_segment_length:
            segments.append(segment.numpy())
        last_end = silence["to"]

    # 最後のセグメントを確認
    if last_end < audio_tensor.shape[-1] / sampling_rate:
        start_sample = int(last_end * sampling_rate)
        segment = audio_tensor[start_sample:]
        if len(segment) / sampling_rate >= min_segment_length:
            segments.append(segment.numpy())

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

def find_closest_segments(audio_path1, audio_path2, sampling_rate=16000, threshold=0.5):
    """
    Find the closest segment pairs between two audio files and return their times.

    Args:
        audio_path1 (str): Path to the first audio file.
        audio_path2 (str): Path to the second audio file.
        sampling_rate (int): Sampling rate of the audio.
        threshold (float): VAD threshold.

    Returns:
        list: List of closest segment pairs with their times.
    """
    segments1 = split_audio_with_silence(audio_path1, sampling_rate=sampling_rate, threshold=threshold)
    segments2 = split_audio_with_silence(audio_path2, sampling_rate=sampling_rate, threshold=threshold)

    closest_pairs = []

    for i, seg1 in enumerate(segments1):
        best_match = None
        best_similarity = -1
        for j, seg2 in enumerate(segments2):
            similarity = calculate_similarity(seg1, seg2, sampling_rate=sampling_rate)
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = (j, similarity)
        if best_match:
            closest_pairs.append((i, best_match[0], best_similarity))

    return closest_pairs

if __name__ == "__main__":
    # 音声ファイルのパス
    source_audio_file = "../data/audio/source/Y1VQdn2pmgo.mp3"
    clip_audio_file = "../data/audio/clipping/qqxKJFDeNHg.mp3"

    # 無音部分での分割と最も近いセグメントの検索
    closest_segments = find_closest_segments(source_audio_file, clip_audio_file)

    # 結果を出力
    for seg1_idx, seg2_idx, similarity in closest_segments:
        print(f"Source Segment {seg1_idx} is closest to Clip Segment {seg2_idx} with similarity {similarity:.4f}")
