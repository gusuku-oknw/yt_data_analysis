import torch
import torchaudio
import librosa
from transformers import Wav2Vec2FeatureExtractor, WavLMForXVector
from sklearn.cluster import AgglomerativeClustering

# モデルと特徴抽出器のロード
model_name = "microsoft/wavlm-large"
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
model = WavLMForXVector.from_pretrained(model_name)

# 音声ファイルの読み込みと前処理
file_path = "path_to_audio_file.wav"
waveform, sample_rate = librosa.load(file_path, sr=16000)
inputs = feature_extractor(waveform, sampling_rate=16000, return_tensors="pt")

# モデルで話者埋め込みを抽出
with torch.no_grad():
    embeddings = model(**inputs).embeddings

# クラスタリングによる話者分離
num_speakers = 2  # 話者数を指定
clustering = AgglomerativeClustering(n_clusters=num_speakers).fit(embeddings.squeeze().numpy())
labels = clustering.labels_

# 結果の表示
print("話者ラベル:", labels)
