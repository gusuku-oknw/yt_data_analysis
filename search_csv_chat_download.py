import time
import pandas as pd
import chat_download
import re
import os
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

# ディレクトリ作成
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
    channel_pattern = re.compile(r'@(?!watch|live|embed)[a-zA-Z0-9_-]+')  # ＠から始まるチャンネルURLを検出

    for url in url_list:
        # 不完全なプロトコルを修正
        if url.startswith("https//"):
            url = url.replace("https//", "https://")
        elif not url.startswith("http"):
            url = "https://" + url

        # クエリパラメータを削除
        if "&t=" in url:
            url = url.split("&t=")[0]  # タイムスタンプパラメータ以前の部分を取得

        # `/channel/` または `＠` が含まれるチャンネルURLをスキップ
        if "/channel/" in url or channel_pattern.search(url):
            continue

        # YouTubeまたはTwitchのURLとして有効かチェック
        if youtube_url_pattern.match(url) or twitch_url_pattern.match(url):
            valid_urls.append(url)

    return valid_urls

# csvファイルtoチャットダウンロード
def list_original_urls(csv_file, base_directory="data/chat_messages", url_column="Original URL", video_url_column="Video URL", delete_multiple=False):
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

        # Video URLとOriginal URLの対応を保持するリストを作成
        urls = df[[video_url_column, url_column]].dropna()

    except FileNotFoundError:
        print(f"CSVファイル '{csv_file}' が見つかりません。")
        return pd.DataFrame()
    except Exception as e:
        print(f"エラーが発生しました: {str(e)}")
        return pd.DataFrame()

    target_directory = create_directory(base_directory)

    all_url = []
    for _, row in urls.iterrows():
        video_url = row[video_url_column]
        original_url = row[url_column]
        split_url_list = split_urls(original_url)

        if delete_multiple and len(split_url_list) > 1:
            continue  # 複数URLを含む行を削除

        for split_url in split_url_list:
            all_url.append({
                "Video URL": video_url,
                "Original URL": split_url
            })

    # フィルタリングとURLの修正
    filtered_all_url = []
    for item in all_url:
        corrected_urls = filter_and_correct_urls([item["Original URL"]])
        for corrected_url in corrected_urls:
            filtered_all_url.append({
                "Video URL": item["Video URL"],
                "Original URL": corrected_url
            })

    # 重複を除去
    unique_urls = []
    seen = set()
    for item in filtered_all_url:
        key = (item["Video URL"], item["Original URL"])
        if key not in seen:
            seen.add(key)
            unique_urls.append(item)

    print(len(unique_urls), "個の有効なURLが見つかりました。")

    download_records = []  # ダウンロード結果を保存するリスト

    for item in tqdm(unique_urls, desc="チャットダウンロード処理"):
        video_url = item["Video URL"]
        original_url = item["Original URL"]

        try:
            video_id = chat_download.get_video_id_from_url(chat_download.remove_query_params(original_url))
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

            print(f"\r処理中のURL: {original_url}")
            if (chat_download.chat_download_csv(original_url, target_directory)) is None:
                print(f"\rエラーが発生しました: ダウンロード失敗 - URL: {original_url}")
                continue

            print(f"\rチャットデータを保存しました: {original_url}")
            download_records.append({
                "Video URL": video_url,
                "Original URL": original_url,
                "File Path": os.path.abspath(file_path)
            })

        except Exception as e:
            if "Private video" in str(e):
                print(f"プライベート動画のためスキップします: {original_url}")
            else:
                print(f"\rエラーが発生しました: {e} - URL: {original_url}")

    print("\nすべての処理が完了しました")

    # ダウンロード結果をDataFrameに変換して返す
    result_df = pd.DataFrame(download_records)
    return result_df


# 実行例
if __name__ == "__main__":
    import pandas as pd

    # CSVファイルのパスを指定
    csv_file = "./data/test_processed.csv"

    # 処理の実行と結果の保存
    result_df = list_original_urls(csv_file, delete_multiple=True)

    # 結果をCSVに保存
    output_csv = "./data/download_results.csv"
    result_df.to_csv(output_csv, index=False, encoding="utf-8-sig")

    print(f"ダウンロード結果を {output_csv} に保存しました。")
