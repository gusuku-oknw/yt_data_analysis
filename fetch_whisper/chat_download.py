from datetime import datetime
from chat_downloader import ChatDownloader
import pandas as pd
import tqdm
import os
from urllib.parse import urlparse, parse_qs
import re
import progressbar
from openai import OpenAI
from dotenv import load_dotenv

# ===== 追加: SQLAlchemyエンジンのインポート =====
from sqlalchemy import create_engine, inspect


class ImageText:
    def __init__(self):
        load_dotenv()
        self.client = OpenAI(api_key=os.environ['OpenAIKey'])

    def image2text(self, image_url):
        """
        画像URLから画像内の文字を抽出する。

        Parameters:
            image_url (str): 画像のURL。

        Returns:
            str: 抽出された文字列、またはエラーメッセージ。
        """
        if not image_url.strip():
            return "None"  # 空白のみの場合

        # OpenAI APIへのプロンプト構築
        message = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "この画像にかかれている文字と感情を抽出してください。"
                            "なかった場合なんと言っていそうですか？わからなければNoneとしてください\n"
                            "[Joy, Sadness, Anticipation, Surprise, Anger, Fear, Disgust, Trust]"
                            "の中で選んでください。テキストのみで出力してください。\n"
                            "例: Hello, World!: Joy\n"
                            "None: Anger"
                            "None: None"  # この行は必須です (最後の行以外にも追加可能
                        )
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_url,
                        },
                    },
                ],
            }
        ]

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=message,
                temperature=0.2,
            )
            extracted = response.choices[0].message.content.strip()
        except Exception as e:
            print("エラーが発生しました:", str(e))
            extracted = "エラーが発生しました。"

        return extracted


# URLアンサンブル
def split_urls(row):
    """
    行に含まれる複数のURLを分割してリストとして返す。チャンネルURLは除外。
    """
    url_pattern = re.compile(r'https?://[^\s,]+')
    urls = url_pattern.findall(row)
    return filter_and_correct_urls(urls, allow_channels=False)

# URLフィルタリング
def filter_and_correct_urls(url_list, allow_playlists=False, allow_channels=False, exclude_twitch=True):
    """
    URLリストをフィルタリングし、不完全なURLを補正して有効なYouTubeおよびTwitch URLのみを返す。

    Parameters:
        url_list (list): URLのリスト。
        allow_playlists (bool): プレイリストURLを許可するかどうか。
        allow_channels (bool): チャンネルURLを許可するかどうか。
        exclude_twitch (bool): TwitchのURLを除外するかどうか。

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

        # Twitch URLを除外する場合
        if exclude_twitch and twitch_url_pattern.match(url):
            continue

        # YouTubeまたはTwitchのURLとして有効かチェック
        if youtube_url_pattern.match(url) or (not exclude_twitch and twitch_url_pattern.match(url)):
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
    """
    parsed_url = urlparse(url)
    query_params = parse_qs(parsed_url.query)

    # v= パラメータが存在する場合はそれを保持
    if "v" in query_params:
        base_url = f"{parsed_url.scheme}://{parsed_url.netloc}{parsed_url.path}"
        return f"{base_url}?v={query_params['v'][0]}"
    else:
        # v= パラメータがない場合は完全にクエリを削除
        return f"{parsed_url.scheme}://{parsed_url.netloc}{parsed_url.path}"


def chat_download_data(url):
    """
    指定されたYouTube URLからチャットデータを取得し、メンバー情報を数値データに変換しDataFrameを返す。

    Parameters:
        url (str): YouTube動画のURL。

    Returns:
        pd.DataFrame: チャットデータ。
    """
    messages_data = []

    try:
        # ChatDownloader を使ってチャットデータを取得
        chat = ChatDownloader().get_chat(url)

        for message in chat:
            time_in_seconds = message.get('time_in_seconds', 'N/A')
            message_text = message.get('message', 'N/A')
            amount = message.get('money', {}).get('amount', 'N/A')

            author = message.get('author', {})
            author_details = {
                "Author Name": author.get('name', 'N/A'),
                "Author ID": author.get('id', 'N/A'),
            }

            # メンバー情報抽出
            badges = author.get('badges', [])
            member_info = 0  # デフォルト0（非メンバー）
            badge_icon_url = ""
            for badge in badges:
                title = badge.get('title', 'N/A')
                icons = badge.get('icons', [])
                if icons:
                    badge_icon_url = icons[0].get('url', '')

                # "Member"が含まれる場合のみ数値変換
                if "Member" in title:
                    match = re.search(r"(\d+)\s*(year|month)", title, re.IGNORECASE)
                    if match:
                        number = int(match.group(1))
                        unit = match.group(2).lower()
                        if unit == "year":
                            member_info = number * 12
                        elif unit == "month":
                            member_info = number

            # スタンプ画像URLの抽出
            stamp_image_url = None
            if 'emotes' in message:
                for emote in message['emotes']:
                    if 'images' in emote:
                        stamp_image_url = emote['images'][0].get('url', None)
                        break

            if not stamp_image_url and 'sticker_images' in message:
                stamp_image_url = message['sticker_images'][0].get('url', None)

            # データをリストに追加
            messages_data.append({
                "Time_in_seconds": time_in_seconds,
                "Message": message_text,
                "Amount": amount,
                **author_details,
                "Member Info (Months)": member_info,
                "Badge Icon URL": badge_icon_url,
                "Stamp Image URL": stamp_image_url if stamp_image_url else "No stamp image"
            })

    except Exception as e:
        print(f"Error during chat download: {e}")
        return None

    # DataFrameに変換して返す
    df = pd.DataFrame(messages_data)

    return df


def message_stamp2text(df, stamp_mapping, image_text_extractor):
    """
    DataFrame内のメッセージとスタンプ画像を処理し、新しいカラムを追加する。

    Parameters:
        df (pd.DataFrame): 元のチャットデータを含むDataFrame。
        stamp_mapping (dict): スタンプ種類と説明のマッピング辞書。
        image_text_extractor (ImageText): スタンプ画像から文字と感情を抽出するクラスのインスタンス。

    Returns:
        pd.DataFrame: 新しいカラムを追加したDataFrame。
    """

    def process_row(row):
        # 元のメッセージを保存
        original_message = row['Message']
        message = original_message.replace('□', '').strip()  # '□'を削除
        stamps = []
        remaining_message = message

        # スタンプを抽出して削除
        while ':_' in remaining_message:
            start_idx = remaining_message.find(':_')
            end_idx = remaining_message.find(':', start_idx + 1)
            if end_idx != -1:
                stamp_code = remaining_message[start_idx + 2:end_idx]
                stamps.append(stamp_code)
                remaining_message = remaining_message[:start_idx] + remaining_message[end_idx + 1:]
            else:
                break

        # スタンプの種類と感情を保存
        stamp_texts = []
        stamp_emotions = []
        for stamp in stamps:
            # スタンプの説明を取得
            stamp_description = stamp_mapping.get(stamp, f"Unknown Stamp: {stamp}")
            # 画像から文字と感情を抽出
            if row['Stamp Image URL'] != "No stamp image":
                extracted_text = image_text_extractor.image2text(row['Stamp Image URL'])
                stamp_text, stamp_emotion = extracted_text.split(": ", 1) if ": " in extracted_text else (
                    extracted_text, "Unknown")
                stamp_texts.append(f"{stamp_description}: {stamp_text}")
                stamp_emotions.append(stamp_emotion)
            else:
                stamp_texts.append(f"{stamp_description}: No image available")
                stamp_emotions.append("Unknown")

        # 最終メッセージ処理
        processed_message = remaining_message.strip()

        return original_message, processed_message, stamps, "; ".join(stamp_texts), "; ".join(stamp_emotions)

    # 新しいカラムを追加
    df[['Original Message', 'Message', 'Stamp Codes', 'Stamp Texts', 'Stamp Emotions']] = df.apply(
        lambda row: pd.Series(process_row(row)), axis=1
    )

    return df


# ===== 新規追加: DataFrameをSQLに保存する関数 =====
def save_to_sql(dataframe, db_url, table_name="chat_data", if_exists="append"):
    """
    指定されたDataFrameをデータベースに保存する。

    Parameters:
        dataframe (pd.DataFrame): 保存するデータ。
        db_url (str): "sqlite:///XXXX.db" などのDB接続URL。
        table_name (str): 保存先テーブル名。
        if_exists (str): テーブルが既に存在する場合の動作（replace, append, fail）。
    """
    # SQLAlchemyのエンジンを作成
    engine = create_engine(db_url)

    # DataFrameをSQLへ保存
    dataframe.to_sql(table_name, con=engine, if_exists=if_exists, index=False)

    print(f"データをデータベース({db_url})のテーブル '{table_name}' に保存しました。")


def list_original_urls(
    csv_file,
    base_directory="../data/chat_messages",
    url_column="Original URL",
    video_url_column="Video URL",
    delete_multiple=False,
    db_url="sqlite:///chat_data.db",       # 保存先のDB URL
):
    """
    指定されたCSVファイルのオリジナルURLカラムからURLを取得し、チャットをダウンロードして
    データベースに保存する。

    Parameters:
        csv_file (str): CSVファイルのパス。
        base_directory (str): チャットメッセージの保存先の基本ディレクトリ（不要なら使わない）。
        url_column (str): URLが記載されているカラム名。
        video_url_column (str): Video URLが記載されているカラム名。
        delete_multiple (bool): URLカラムに複数のURLがある場合、行を削除するかどうか。
        db_url (str): データベース接続URL (例: "sqlite:///chat_data.db")。

    Returns:
        pd.DataFrame: Video URL、Original URL、保存先テーブル名などの対応表をDataFrameで返す。
    """
    try:
        df = pd.read_csv(csv_file, encoding="utf-8-sig")

        # カラム名の動的選択
        if url_column not in df.columns and "Original videoURL" in df.columns:
            url_column = "Original videoURL"

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
    bar = progressbar.ProgressBar(max_value=len(filtered_all_urls))  # プログレスバーを設定

    # SQLAlchemyエンジンとインスペクターを作成
    engine = create_engine(db_url)
    inspector = inspect(engine)

    for i, item in enumerate(filtered_all_urls):
        video_url = item["Video URL"]
        original_url = item["Original URL"]

        try:
            # 動画IDの取得
            video_id = get_video_id_from_url(remove_query_params(original_url))

            # テーブルの存在をチェック
            if video_id in inspector.get_table_names():
                print(f"既存テーブルが見つかりました。スキップします: {video_id}")
                download_records.append({
                    "Video URL": video_url,
                    "Original URL": original_url,
                    "DB Table": video_id,
                    "Video ID": video_id,
                    "Status": "Skipped (Already Exists)"
                })
                bar.update(i + 1)
                continue

            # 新規ダウンロードと保存
            df_chat = chat_download_data(original_url)

            if df_chat is not None and not df_chat.empty:
                # ====== SQLへ保存 ======
                save_to_sql(df_chat, db_url=db_url, table_name=video_id, if_exists="append")

                download_records.append({
                    "Video URL": video_url,
                    "Original URL": original_url,
                    "DB Table": video_id,
                    "Video ID": video_id,
                    "Status": "Downloaded"
                })
        except Exception as e:
            print(f"エラーが発生しました: {e} - URL: {original_url}")
        bar.update(i + 1)

    print("\nすべての処理が完了しました")

    return pd.DataFrame(download_records)


if __name__ == "__main__":
    # # 単体テスト的に呼び出す場合
    # # 例: 指定URLのチャットを取得し、chat_data.db の chat_data テーブルに保存
    # df_test = chat_download_data("https://www.youtube.com/watch?v=a4KN-5n0YF0")
    # if df_test is not None and not df_test.empty:
    #     save_to_sql(df_test, db_url="sqlite:///chat_data.db", table_name="chat_data_sample")

    # 実運用例: CSVファイルに含まれるURLを一括で取り込み、SQLに保存
    csv_file = "../data/にじさんじ　切り抜き_20250102_202807.csv"
    result_df = list_original_urls(
        csv_file,
        delete_multiple=True,
        url_column="Original videoURL",
        db_url="sqlite:///chat_data.db",
    )
    result_df.to_csv("../data/download_results.csv", index=False, encoding="utf-8-sig")
