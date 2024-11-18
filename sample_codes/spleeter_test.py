import subprocess
import os

def extract_vocals(audio_file, output_dir="output"):
    command = ["demucs", "-d", "cuda", "-o", output_dir, audio_file]
    subprocess.run(command, check=True)

    basename = os.path.splitext(os.path.basename(audio_file))[0]

    return f"{output_dir}/htdemucs/{basename}/vocals.wav"


audio_path = '../data/sound/clipping_audio/7-1fNxXj_xM.mp3'
# ディレクトリのルートを取得
root_directory = os.path.dirname(audio_path)

print(extract_vocals(audio_path, root_directory))
