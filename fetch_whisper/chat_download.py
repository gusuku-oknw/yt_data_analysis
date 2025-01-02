# これはサンプルの Python スクリプトです。
from datetime import datetime
from chat_downloader import ChatDownloader
import pandas as pd
import tqdm
import os
from urllib.parse import urlparse, parse_qs
from FunctionUtils import FunctionUtils
import re
from tqdm import tqdm

# URLアンサンブル
def split_urls(row):
    """
    行に含まれる複数のURLを分割してリストとして返す。
    """
    url_pattern = re.compile(
        r'https?://[^\s,]+'
    )
    return url_pattern.findall(row)

def create_directory(base_directory):
    """
    保存先ディレクトリを作成。

    Parameters:
        base_directory (str): 基本のディレクトリパス。

    Returns:
        str: 作成したディレクトリのパス。
    """
    os.makedirs(base_directory, exist_ok=True)
    return base_directory

# URLフィルタリング
def filter_and_correct_urls(url_list, allow_playlists=False, allow_channels=False):
    """
    URLリストをフィルタリングし、不完全なURLを補正して有効なYouTubeおよびTwitch URLのみを返す。

    Parameters:
        url_list (list): URLのリスト。
        allow_playlists (bool): プレイリストURLを許可するかどうか。
        allow_channels (bool): チャンネルURLを許可するかどうか。

    Returns:
        list: フィルタリングされた有効なURLのリスト。
    """
    valid_urls = []
    youtube_url_pattern = re.compile(
        r'(https://)?(www\.)?(youtube\.com|youtu\.be)/(watch\?v=|live/|embed/|[a-zA-Z0-9_-]+)'
    )
    twitch_url_pattern = re.compile(
        r'(https://)?(www\.)?twitch\.tv/videos/\d+'
    )
    playlist_pattern = re.compile(r'list=')
    channel_pattern = re.compile(r'@(?!watch|live|embed)[a-zA-Z0-9_-]+')

    for url in url_list:
        # 不完全なプロトコルを修正
        if url.startswith("https//"):
            url = url.replace("https//", "https://")
        elif not url.startswith("http"):
            url = "https://" + url

        # クエリパラメータを削除
        if "&t=" in url:
            url = url.split("&t=")[0]

        # プレイリストを許可しない場合、プレイリストURLをスキップ
        if not allow_playlists and playlist_pattern.search(url):
            continue

        # チャンネルURLを許可しない場合、チャンネルURLをスキップ
        if not allow_channels and ("/channel/" in url or channel_pattern.search(url)):
            continue

        # YouTubeまたはTwitchのURLとして有効かチェック
        if youtube_url_pattern.match(url) or twitch_url_pattern.match(url):
            valid_urls.append(url)

    return valid_urls

# URLから動画IDを取得
def get_video_id_from_url(url):
    """
    URLからYouTube動画IDを抽出する。
    - 通常のYouTube URL (例: https://www.youtube.com/watch?v=...)
    - 短縮URL (例: https://youtu.be/...)
    - ライブ配信URL (例: https://www.youtube.com/live/...)
    - プレイリスト付きURL (例: https://www.youtube.com/watch?v=...&list=...)

    Parameters:
        url (str): 処理対象のURL。

    Returns:
        str: 抽出した動画ID。抽出できない場合は 'unknown_video_id' を返す。
    """
    parsed_url = urlparse(url)

    # 1. 通常のYouTube動画URL (例: https://www.youtube.com/watch?v=...)
    if "youtube.com" in parsed_url.netloc:
        query_params = parse_qs(parsed_url.query)
        if "v" in query_params:  # v= パラメータが存在する場合
            return query_params["v"][0]
        # /live/形式のライブ配信URLの場合
        if "/live/" in parsed_url.path:
            return parsed_url.path.split("/")[-1]

    # 2. 短縮URL (例: https://youtu.be/...)
    if "youtu.be" in parsed_url.netloc:
        return parsed_url.path.strip("/")  # 短縮URLの場合、パス部分をIDとして返す

    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    return f"{current_time}_unknown_video_id"  # 未対応の形式の場合


def remove_query_params(url):
    """
    URLから不要なクエリパラメータを削除する。ただし、YouTubeの動画IDを示す "v=" パラメータは保持する。

    Parameters:
        url (str): 処理対象のURL。

    Returns:
        str: 修正されたURL。

    詳細:
        - URLを解析してクエリパラメータを抽出します。
        - "v=" パラメータが存在する場合はそれを保持してURLを再構築します。
        - "v=" パラメータが存在しない場合は、クエリ部分を完全に削除します。
    """
    parsed_url = urlparse(url)
    query_params = parse_qs(parsed_url.query)

    # v= パラメータが存在する場合は保持
    if "v" in query_params:
        base_url = f"{parsed_url.scheme}://{parsed_url.netloc}{parsed_url.path}"
        return f"{base_url}?v={query_params['v'][0]}"
    else:
        # v= パラメータがない場合は完全にクエリを削除
        return f"{parsed_url.scheme}://{parsed_url.netloc}{parsed_url.path}"


def chat_download(url):
    # 抽出データのリスト
    messages_data = []

    try:
        chat = ChatDownloader().get_chat(url,
                                         # start_time='0:00',
                                         # end_time='0:10:00',
                                         message_groups=['messages', 'superchat']
                                         )
    # create a generator
    except Exception as e:
        print(f"Error during transcription: {e}")
        return None

    # --------------------------------------------------------------------
    # chat = "<chat_downloader.sites.common.Chat object at 0x0000027CC147BDF0>"
    for message in tqdm.tqdm(chat):  # iterate over messages

        # print(f'message: {message}')
        # 各メッセージからデータを抽出
        message_text = message.get('message')
        amount = message.get('money', {}).get('amount')
        time_in_seconds = message.get('time_in_seconds')

        # 必要なデータを辞書形式でリストに追加
        messages_data.append({
            "Time_in_seconds": time_in_seconds,
            "Message": message_text,
            "Amount": amount,
        })
    return pd.DataFrame(messages_data)

def list_original_urls(csv_file, base_directory="../data/chat_messages", url_column="Original URL", video_url_column="Video URL", delete_multiple=False):
    """
    指定されたCSVファイルのオリジナルURLカラムからURLを取得し、チャットをダウンロードする。

    Parameters:
        csv_file (str): CSVファイルのパス。
        base_directory (str): チャットメッセージの保存先の基本ディレクトリ。
        url_column (str): URLが記載されているカラム名。
        video_url_column (str): Video URLが記載されているカラム名。
        delete_multiple (bool): URLカラムに複数のURLがある場合、行を削除するかどうか。

    Returns:
        pd.DataFrame: Video URL、Original URL、保存ファイルパスの対応表。
    """
    try:
        df = pd.read_csv(csv_file, encoding="utf-8-sig")

        if url_column not in df.columns or video_url_column not in df.columns:
            print(f"指定されたカラム '{url_column}' または '{video_url_column}' がCSVファイルに存在しません。")
            return pd.DataFrame()

        urls = df[[video_url_column, url_column]].dropna()
    except FileNotFoundError:
        print(f"CSVファイル '{csv_file}' が見つかりません。")
        return pd.DataFrame()
    except Exception as e:
        print(f"エラーが発生しました: {str(e)}")
        return pd.DataFrame()

    target_directory = create_directory(base_directory)

    all_urls = []
    for _, row in urls.iterrows():
        video_url = row[video_url_column]
        original_url = row[url_column]
        split_url_list = split_urls(original_url)

        if delete_multiple and len(split_url_list) > 1:
            continue

        for split_url in split_url_list:
            all_urls.append({
                "Video URL": video_url,
                "Original URL": split_url
            })

    filtered_all_urls = []
    for item in all_urls:
        corrected_urls = filter_and_correct_urls([item["Original URL"]])
        for corrected_url in corrected_urls:
            filtered_all_urls.append({
                "Video URL": item["Video URL"],
                "Original URL": corrected_url
            })

    download_records = []
    for item in tqdm(filtered_all_urls, desc="チャットダウンロード処理"):
        video_url = item["Video URL"]
        original_url = item["Original URL"]

        try:
            video_id = re.search(r'(?<=v=)[^&]+', original_url).group() or "unknown_video_id"
            file_name = f"{video_id}.csv"
            file_path = os.path.join(target_directory, file_name)

            if os.path.exists(file_path):
                print(f"\rファイルが既に存在します。スキップします: {file_path}")
                download_records.append({
                    "Video URL": video_url,
                    "Original URL": original_url,
                    "File Path": os.path.abspath(file_path)
                })
                continue

            df = chat_download(original_url)
            if df is not None:
                df.to_csv(file_path, index=False, encoding="utf-8-sig")
                download_records.append({
                    "Video URL": video_url,
                    "Original URL": original_url,
                    "File Path": os.path.abspath(file_path)
                })
        except Exception as e:
            print(f"エラーが発生しました: {e} - URL: {original_url}")

    print("\nすべての処理が完了しました")
    return pd.DataFrame(download_records)


if __name__ == "__main__":
    csv_file = "../data/test_processed.csv"
    result_df = list_original_urls(csv_file, delete_multiple=True)
    output_csv = "../data/download_results.csv"
    result_df.to_csv(output_csv, index=False, encoding="utf-8-sig")
    print(f"ダウンロード結果を {output_csv} に保存しました。")
