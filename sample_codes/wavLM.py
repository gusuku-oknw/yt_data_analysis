import subprocess

from transformers import Wav2Vec2FeatureExtractor, WavLMForXVector
import torch
import librosa
import os
import logging
from itertools import product
from datetime import timedelta
from tqdm import tqdm

# モデル名
model_name = "microsoft/wavlm-large"

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

def detect_silence(audio_tensor, sampling_rate=16000, threshold=0.5):
    """
    Silero VADを使用して無音部分を検出します。

    Args:
        audio_tensor (torch.Tensor): 音声データ。
        sampling_rate (int): サンプリングレート。
        threshold (float): VADのしきい値。

    Returns:
        list: 無音部分のリスト（開始時刻と終了時刻）。
    """
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

def split_audio_with_silence(audio_tensor, sampling_rate=16000, threshold=0.5, min_segment_length=3.0):
    """
    音声を無音部分で分割し、最小長さを満たすセグメントのみを保持します。

    Args:
        audio_tensor (torch.Tensor): 音声データ。
        sampling_rate (int): サンプリングレート。
        threshold (float): VADのしきい値。
        min_segment_length (float): 最小セグメント長（秒）。

    Returns:
        list: 分割された音声セグメント。
    """
    silences = detect_silence(audio_tensor, sampling_rate=sampling_rate, threshold=threshold)
    segments = []
    segment_times = []

    last_end = 0
    temp_segment = []
    temp_start_time = 0

    for silence in tqdm(silences):
        start_sample = int(last_end * sampling_rate)
        end_sample = int(silence["from"] * sampling_rate)
        segment = audio_tensor[start_sample:end_sample]

        # 一時セグメントに追加
        temp_segment.append(segment)
        total_length = sum([seg.shape[0] for seg in temp_segment]) / sampling_rate

        # セグメント長が十分か確認
        if total_length >= min_segment_length:
            merged_segment = torch.cat(temp_segment)
            segments.append(merged_segment.numpy())
            segment_times.append((temp_start_time, last_end))
            temp_segment = []
            temp_start_time = silence["to"]

        last_end = silence["to"]

    # 残りのセグメントを確認し、次のセグメントと統合
    if temp_segment:
        merged_segment = torch.cat(temp_segment)
        if segments and len(segments[-1]) / sampling_rate < min_segment_length:
            segments[-1] = torch.cat([torch.tensor(segments[-1]), merged_segment]).numpy()
            segment_times[-1] = (segment_times[-1][0], last_end)
        else:
            segments.append(merged_segment.numpy())
            segment_times.append((temp_start_time, last_end))

    if last_end < len(audio_tensor) / sampling_rate:
        start_sample = int(last_end * sampling_rate)
        segment = audio_tensor[start_sample:]
        if len(segment) / sampling_rate >= min_segment_length:
            segments.append(segment.numpy())
            segment_times.append((last_end, len(audio_tensor) / sampling_rate))
        else:
            if segments:
                segments[-1] = torch.cat([torch.tensor(segments[-1]), segment]).numpy()
                segment_times[-1] = (segment_times[-1][0], len(audio_tensor) / sampling_rate)
            else:
                segments.append(segment.numpy())
                segment_times.append((last_end, len(audio_tensor) / sampling_rate))

    return segments, segment_times

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

def find_closest_segments_continuous(audio1, audio2, sampling_rate=16000, threshold=0.5):
    """
    Find the closest segment pairs between two audio files with continuity in matching.

    Args:
        audio1 (torch.Tensor): 1つ目の音声データ。
        audio2 (torch.Tensor): 2つ目の音声データ。
        sampling_rate (int): サンプリングレート。
        threshold (float): VADのしきい値。

    Returns:
        list: もっとも近いセグメントペア。
    """
    segments1, times1 = split_audio_with_silence(audio1, sampling_rate=sampling_rate, threshold=threshold)
    segments2, times2 = split_audio_with_silence(audio2, sampling_rate=sampling_rate, threshold=threshold)

    closest_pairs = []

    for j, (seg2, time2) in tqdm(enumerate(zip(segments2, times2))):
        best_match = None
        best_similarity = -1

        for i, (seg1, time1) in enumerate(zip(segments1, times1)):
            similarity = calculate_similarity(seg1, seg2, sampling_rate=sampling_rate)
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = (i, similarity)

        if best_match and best_similarity >= 0.90:
            closest_pairs.append((best_match[0], j, best_match[1], times1[best_match[0]], time2))

    return closest_pairs

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

def format_time(seconds):
    """
    秒を時間:分:秒の形式に変換します。

    Args:
        seconds (float): 秒数。

    Returns:
        str: 時間:分:秒形式の文字列。
    """
    return str(timedelta(seconds=round(seconds)))


if __name__ == "__main__":
    # 音声ファイルのパス
    source_audio_file = "../data/audio/source/pnHdRQbR2zs.mp3"
    clip_audio_file = "../data/audio/clipping/-bRcKCM5_3E.mp3"

    clip_audio_file = extract_vocals(clip_audio_file)

    # サンプリングレートを指定してロード
    source_audio, source_sr = librosa.load(source_audio_file, sr=16000)
    clip_audio, clip_sr = librosa.load(clip_audio_file, sr=16000)

    # サンプリングレートの一致確認
    assert source_sr == clip_sr, "Source and clip audio files must have the same sampling rate!"

    # 無音部分での分割と最も近いセグメントの検索
    closest_segments = find_closest_segments_continuous(torch.tensor(source_audio), torch.tensor(clip_audio), sampling_rate=source_sr)

    # 結果を出力
    for seg1_idx, seg2_idx, similarity, time1, time2 in closest_segments:
        print(f"Clip Segment {seg2_idx} (time: {format_time(time2[0])}-{format_time(time2[1])}) is closest to Source Segment {seg1_idx} (time: {format_time(time1[0])}-{format_time(time1[1])}) with similarity {similarity:.4f}")
