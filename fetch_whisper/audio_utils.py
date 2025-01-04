import os
from yt_dlp import YoutubeDL
import subprocess
import torch
from yt_url_utils import YTURLUtils
yt_utils = YTURLUtils()


# YouTube動画または音声をダウンロード
def download_yt_sound(url, output_dir="../data/sound", audio_only=True):
    """
    YouTube動画または音声をダウンロード。

    Parameters:
        url (str): ダウンロード対象のURL。
        output_dir (str): 出力ディレクトリ。
        audio_only (bool): Trueの場合、音声のみをダウンロード。

    Returns:
        str: ダウンロードされたファイルのパス。
    """
    os.makedirs(output_dir, exist_ok=True)

    # 出力ファイル名を生成（拡張子なし）
    video_id = yt_utils.get_video_id_from_url(yt_utils.remove_query_params(url))
    file_name = video_id  # 拡張子はFFmpegで付加
    file_path_no_ext = os.path.join(output_dir, file_name)
    file_path_with_ext = f"{file_path_no_ext}.mp3"

    # ファイルが存在する場合はダウンロードをスキップ
    if os.path.exists(file_path_with_ext):
        print(f"ファイルは既に存在します: {file_path_with_ext}")
        return file_path_with_ext

    ydl_opts = {
        'format': 'bestaudio' if audio_only else 'bestvideo+bestaudio',
        'outtmpl': file_path_no_ext,  # 拡張子を付けない
        'noplaylist': True,
        'quiet': False,
        'postprocessors': [  # 音声のみの場合の後処理
            {
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': '192',
            }
        ] if audio_only else None
    }

    # YoutubeDLを使用してダウンロード
    with YoutubeDL(ydl_opts) as ydl:
        ydl.extract_info(url, download=True)

    print(f"ファイルをダウンロードしました: {file_path_with_ext}")
    return file_path_with_ext

# demucsを使用してボーカルを抽出
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

        command = ['demucs', '-d', device, '-o', root_directory, audio_file, '--two-stems STEM']
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

if __name__ == "__main__":
    # テスト用のURL
    url = "https://www.youtube.com/watch?v=WJoNiYYxgz8"

    # ダウンロード
    # source_path = download_yt_sound(url, output_dir="../data/audio")
    clip_path = download_yt_sound(url, output_dir="../data/audio")

    # ボーカル抽出
    clip_path = extract_vocals(clip_path)
    print("処理が完了しました。")