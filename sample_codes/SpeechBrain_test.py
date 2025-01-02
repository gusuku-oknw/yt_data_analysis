from speechbrain.inference import SpeakerDiarization
import torch

# モデルのロード
diarization = SpeakerDiarization.from_hparams(
    source="speechbrain/ami_diarization",
    savedir="pretrained_models/diarization"
)


# 入力音声ファイルのパス
audio_file = "../data/audio/source/bh4ObBry9q4.mp3"

# 話者分離の実行
diarization_output = diarization(audio_file)

# 結果の表示
print("Speaker Diarization Output:")
print(diarization_output)

# セグメントごとに話者を表示
for segment, speaker in diarization_output.items():
    start_time, end_time = segment
    print(f"Speaker {speaker}: {start_time:.2f}s - {end_time:.2f}s")
