import pandas as pd
import chat_download
import re
import os

def split_urls(row):
    """
    行に含まれる複数のURLを分割してリストとして返す。
    """
    url_pattern = re.compile(
        r'https?://[^\s,]+'
    )
    return url_pattern.findall(row)


def create_directory(base_directory, csv_filename):
    """
    CSVファイル名を基に保存先ディレクトリを作成。

    Parameters:
        base_directory (str): 基本のディレクトリパス。
        csv_filename (str): CSVファイル名。

    Returns:
        str: 作成したディレクトリのパス。
    """
    # 拡張子を除いたファイル名をディレクトリ名として使用
    directory_name = os.path.splitext(os.path.basename(csv_filename))[0]
    target_directory = os.path.join(base_directory, directory_name)

    # ディレクトリを作成（存在しない場合のみ）
    os.makedirs(target_directory, exist_ok=True)
    return target_directory


def filter_and_correct_urls(url_list):
    """
    URLリストをフィルタリングし、不完全なURLを補正して有効なYouTubeおよびTwitch URLのみを返す。
    """
    valid_urls = []
    youtube_url_pattern = re.compile(
        r'(https://)?(www\.)?(youtube\.com|youtu\.be)/(watch\?v=|live/|embed/|[a-zA-Z0-9_-]+)'
    )
    twitch_url_pattern = re.compile(
        r'(https://)?(www\.)?twitch\.tv/videos/\d+'
    )

    for url in url_list:
        # 不完全なプロトコルを修正
        if url.startswith("https//"):
            url = url.replace("https//", "https://")
        elif not url.startswith("http"):
            url = "https://" + url

        # クエリパラメータを削除
        if "&t=" in url:
            url = url.split("&t=")[0]  # タイムスタンプパラメータ以前の部分を取得

        # `/channel/` が含まれる場合はスキップ
        if "/channel/" in url:
            continue

        # YouTubeまたはTwitchのURLとして有効かチェック
        if youtube_url_pattern.match(url) or twitch_url_pattern.match(url):
            valid_urls.append(url)

    return valid_urls


def list_original_urls(csv_file, base_directory="data/chat_messages", url_column="Original URL"):
    """
    指定されたCSVファイルのオリジナルURLカラムからURLを取得し、チャットをダウンロードする。

    Parameters:
        csv_file (str): CSVファイルのパス。
        base_directory (str): チャットメッセージの保存先の基本ディレクトリ。
        url_column (str): URLが記載されているカラム名（デフォルトは "Original videoURL"）。
    """
    urls = []
    try:
        # CSVファイルを読み込む
        df = pd.read_csv(csv_file, encoding="utf-8-sig")  # 2行目以降を読み込む

        # 指定されたカラムが存在するか確認
        if url_column not in df.columns:
            print(f"指定されたカラム '{url_column}' がCSVファイルに存在しません。")
            return

        # URLのリストを取得
        urls = df[url_column].dropna().unique()  # 重複と欠損値を排除

    except FileNotFoundError:
        print(f"CSVファイル '{csv_file}' が見つかりません。")
        return
    except Exception as e:
        print(f"エラーが発生しました: {str(e)}")
        return

    all_url = []
    # URLを表示
    print("オリジナルURLのリスト:")
    for url in urls:
        # 分割されたURLを収集
        split_url_list = split_urls(url)
        all_url.extend(split_url_list)

    # フィルタリングと補正されたURLを処理
    valid_urls = filter_and_correct_urls(all_url)

    # 保存先ディレクトリを作成
    target_directory = create_directory(base_directory, csv_file.split('_', 1)[0])

    valid_urls = list(set(valid_urls))  # 重複を除去してリストに変換

    for valid_url in valid_urls:
        # 動画IDを取得してファイル名を生成
        video_id = chat_download.get_video_id_from_url(chat_download.remove_query_params(valid_url))
        file_name = f"{video_id}.csv"  # 保存するファイル名
        file_path = os.path.join(target_directory, file_name)

        # ファイルが既に存在する場合はスキップ
        if os.path.exists(file_path):
            print(f"ファイルが既に存在します。スキップします: {file_path}")
            continue

        # チャット取得処理を実行
        try:
            print(f"処理中のURL: {valid_url}")
            chat_download.chat_download_csv(valid_url, target_directory)
            print(f"チャットデータを保存しました: {valid_url}")
        except Exception as e:
            print(f"エラーが発生しました: {e} - URL: {valid_url}")


# 実行例
if __name__ == "__main__":
    csv_file = "./data/ホロライブ　切り抜き_2024-11-16_18-26-28_videos_processed.csv"  # 先程のCSVファイルのパス
    print(csv_file.split('_', 1)[0])
    list_original_urls(csv_file)
