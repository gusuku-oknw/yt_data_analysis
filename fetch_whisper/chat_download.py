from datetime import datetime
from chat_downloader import ChatDownloader
import pandas as pd
import os
from urllib.parse import urlparse, parse_qs
import re
import progressbar
from openai import OpenAI
from dotenv import load_dotenv

# ===== 追加: SQLAlchemyエンジンのインポート =====
from sqlalchemy import create_engine, inspect


# --------------------------------------------------------------
# 1) ImageText クラス定義
# --------------------------------------------------------------
class ImageText:
    def __init__(self):
        load_dotenv()
        self.client = OpenAI(api_key=os.environ['OpenAIKey'])

    def image2text(self, image_url):
        """
        画像URLから画像内の文字を抽出する。
        """
        if not image_url.strip():
            return "None"  # 空白のみの場合

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
                            "None: None"
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


# --------------------------------------------------------------
# 2) URL関連のユーティリティ
# --------------------------------------------------------------
def split_urls(row):
    """
    複数URLを分割してリストとして返す。チャンネルURLは除外。
    """
    url_pattern = re.compile(r'https?://[^\s,]+')
    urls = url_pattern.findall(row)
    return filter_and_correct_urls(urls, allow_channels=False)


def filter_and_correct_urls(url_list, allow_playlists=False, allow_channels=False, exclude_twitch=True):
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
    playlist_pattern = re.compile(r'list=')
    channel_pattern = re.compile(r'@(?!watch|live|embed)[a-zA-Z0-9_-]+')

    for url in url_list:
        if url.startswith("https//"):
            url = url.replace("https//", "https://")
        elif not url.startswith("http"):
            url = "https://" + url

        if "&t=" in url:
            url = url.split("&t=")[0]

        if not allow_playlists and playlist_pattern.search(url):
            continue
        if not allow_channels and ("/channel/" in url or channel_pattern.search(url)):
            continue
        if exclude_twitch and twitch_url_pattern.match(url):
            continue

        if youtube_url_pattern.match(url) or (not exclude_twitch and twitch_url_pattern.match(url)):
            valid_urls.append(url)
    return valid_urls


def get_video_id_from_url(url):
    """
    URLからYouTube動画IDを抽出する。
    """
    parsed_url = urlparse(url)
    if "youtube.com" in parsed_url.netloc:
        query_params = parse_qs(parsed_url.query)
        if "v" in query_params:
            return query_params["v"][0]
        if "/live/" in parsed_url.path:
            return parsed_url.path.split("/")[-1]
    if "youtu.be" in parsed_url.netloc:
        return parsed_url.path.strip("/")
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return f"{current_time}_unknown_video_id"


def remove_query_params(url):
    """
    URLから不要なクエリパラメータを削除する。ただし "v=" パラメータは保持。
    """
    parsed_url = urlparse(url)
    query_params = parse_qs(parsed_url.query)
    if "v" in query_params:
        base_url = f"{parsed_url.scheme}://{parsed_url.netloc}{parsed_url.path}"
        return f"{base_url}?v={query_params['v'][0]}"
    else:
        return f"{parsed_url.scheme}://{parsed_url.netloc}{parsed_url.path}"


# --------------------------------------------------------------
# 3) チャットダウンロード
# --------------------------------------------------------------
def chat_download_data(url):
    """
    YouTube URLからチャットデータを取得し、DataFrameで返す。
    """
    messages_data = []
    try:
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

            # メンバー情報
            badges = author.get('badges', [])
            member_info = 0
            badge_icon_url = ""
            for badge in badges:
                title = badge.get('title', 'N/A')
                icons = badge.get('icons', [])
                if icons:
                    badge_icon_url = icons[0].get('url', '')
                if "Member" in title:
                    match = re.search(r"(\d+)\s*(year|month)", title, re.IGNORECASE)
                    if match:
                        number = int(match.group(1))
                        unit = match.group(2).lower()
                        if unit == "year":
                            member_info = number * 12
                        elif unit == "month":
                            member_info = number

            # スタンプ画像
            stamp_image_url = None
            if 'emotes' in message:
                for emote in message['emotes']:
                    if 'images' in emote:
                        stamp_image_url = emote['images'][0].get('url', None)
                        break
            if not stamp_image_url and 'sticker_images' in message:
                stamp_image_url = message['sticker_images'][0].get('url', None)

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

    if not messages_data:
        return None

    df = pd.DataFrame(messages_data)
    return df


import json
from datetime import datetime

# --------------------------------------------------------------
# 4) チャットのスタンプを解析しつつ、Stamp_data 用の情報を収集
# --------------------------------------------------------------
def message_stamp2text(df, stamp_mapping, channel_id):
    """
    チャットのメッセージを解析し、スタンプコード・スタンプ説明などを抽出する。
    - df: チャットデータ
    - stamp_mapping: 既知のスタンプコードマッピング
    - channel_id: どのチャンネルか（Stamp_data に保存するときに使用する）

    Returns:
        df (pd.DataFrame): 新しいカラムを付与したチャットDataFrame
        updated_stamp_mapping (dict): 更新されたスタンプマッピング
        stamps_data_list (list[dict]): Stamp_data 用に格納するレコードのリスト
    """
    image_text_extractor = ImageText()

    # Stamp_data用レコード格納用
    stamps_data_list = []

    def process_row(row):
        original_message = row['Message']
        message = original_message.replace('□', '').strip()
        stamps = []
        remaining_message = message

        # スタンプコード(':_xxx:') 抽出
        while ':_' in remaining_message:
            start_idx = remaining_message.find(':_')
            end_idx = remaining_message.find(':', start_idx + 1)
            if end_idx != -1:
                stamp_code = remaining_message[start_idx + 2:end_idx]
                stamps.append(stamp_code)
                remaining_message = (
                        remaining_message[:start_idx] + remaining_message[end_idx + 1:]
                )
            else:
                break

        stamp_texts = []
        stamp_emotions = []
        stamps_in_this_row = {}

        for stamp in stamps:
            stamp_description = stamp_mapping.get(stamp, f"Unknown Stamp: {stamp}")

            if stamp in stamps_in_this_row:
                reuse_text, reuse_emotion = stamps_in_this_row[stamp]
                stamp_texts.append(f"{stamp_description}: {reuse_text}")
                stamp_emotions.append(reuse_emotion)
                continue

            # 既に確定した情報があれば再利用
            if stamp in stamp_mapping and not stamp_mapping[stamp].startswith("Unknown Stamp:"):
                known_text_emotion = stamp_mapping[stamp]
                if ": " in known_text_emotion:
                    known_text, known_emotion = known_text_emotion.split(": ", 1)
                else:
                    known_text, known_emotion = known_text_emotion, "Unknown"

                stamps_in_this_row[stamp] = (known_text, known_emotion)
                stamp_texts.append(f"{stamp_description}: {known_text}")
                stamp_emotions.append(known_emotion)

                # Stamp_data に追加
                stamps_data_list.append({
                    "channel_id": channel_id,
                    "stamp_code": stamp,
                    "stamp_text": known_text,
                    "stamp_emotion": known_emotion,
                    "created_at": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                })

                continue

            # 未確定 => 画像があればAPI
            if row['Stamp Image URL'] != "No stamp image":
                extracted_text = image_text_extractor.image2text(row['Stamp Image URL'])
                if ": " in extracted_text:
                    stamp_text, stamp_emotion = extracted_text.split(": ", 1)
                else:
                    stamp_text, stamp_emotion = extracted_text, "Unknown"

                if stamp_description.startswith("Unknown Stamp:"):
                    stamp_mapping[stamp] = f"{stamp_text}: {stamp_emotion}"

                stamp_texts.append(f"{stamp_description}: {stamp_text}")
                stamp_emotions.append(stamp_emotion)
                stamps_in_this_row[stamp] = (stamp_text, stamp_emotion)

                # Stamp_data に追加
                stamps_data_list.append({
                    "channel_id": channel_id,
                    "stamp_code": stamp,
                    "stamp_text": stamp_text,
                    "stamp_emotion": stamp_emotion,
                    "created_at": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                })
            else:
                # 画像なし
                stamp_text = "No image available"
                stamp_emotion = "Unknown"
                stamp_texts.append(f"{stamp_description}: {stamp_text}")
                stamp_emotions.append(stamp_emotion)
                stamps_in_this_row[stamp] = (stamp_text, stamp_emotion)

                # Stamp_data に追加
                stamps_data_list.append({
                    "channel_id": channel_id,
                    "stamp_code": stamp,
                    "stamp_text": stamp_text,
                    "stamp_emotion": stamp_emotion,
                    "created_at": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                })

        # `stamps` を JSON 形式で保存する
        processed_message = remaining_message.strip()
        return (
            original_message,
            processed_message,
            json.dumps(stamps, ensure_ascii=False),  # JSON形式で保存
            "; ".join(stamp_texts),
            "; ".join(stamp_emotions),
        )

    df[[
        'Original Message',
        'Message',
        'Stamp Codes',
        'Stamp Texts',
        'Stamp Emotions'
    ]] = df.apply(lambda row: pd.Series(process_row(row)), axis=1)

    return df, stamp_mapping, stamps_data_list


# --------------------------------------------------------------
# 5) DBへ保存する関数群
# --------------------------------------------------------------
def save_to_sql(dataframe, db_url, table_name, if_exists="append"):
    """
    汎用の DataFrame → SQL 保存関数
    """
    engine = create_engine(db_url)
    dataframe.to_sql(table_name, con=engine, if_exists=if_exists, index=False)
    print(f"[SAVE] {table_name} に {len(dataframe)} 件のレコードを追加")


def save_channel_to_sql(channel_data_list, db_url):
    """
    Channel 情報を "Channel_id" テーブルに保存する
    """
    if not channel_data_list:
        return
    df = pd.DataFrame(channel_data_list)
    save_to_sql(df, db_url, table_name="Channel_id", if_exists="append")


def save_stamps_to_sql(stamps_data_list, db_url):
    """
    スタンプ情報を "Stamp_data" テーブルに保存する
    """
    if not stamps_data_list:
        return
    df_stamps = pd.DataFrame(stamps_data_list)
    save_to_sql(df_stamps, db_url, table_name="Stamp_data", if_exists="append")


# --------------------------------------------------------------
# 6) メインの処理
# --------------------------------------------------------------
def list_original_urls(
        csv_file,
        url_column="Original URL",
        video_url_column="Video URL",
        delete_multiple=False,
        db_url="sqlite:///chat_data.db",
):
    """
    CSV のオリジナルURLからチャットを取得し、DBに保存する。
      - 動画チャット: テーブル名は "video_id" (YouTubeのID)
      - チャンネル情報: テーブル名 "Channel_id"
      - スタンプ情報: テーブル名 "Stamp_data"
    """
    # search_yt インスタンスを仮定（実際にはユーザ環境に応じて実装してください）
    from search_yt import search_yt
    search_data = search_yt()

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

    # すべてのURLを集約
    all_urls = []
    for _, row in urls.iterrows():
        video_url = row[video_url_column]
        original_url = row[url_column]
        split_url_list = split_urls(original_url)
        if delete_multiple and len(split_url_list) > 1:
            # URLが複数ある行は無視
            continue
        for split_url in split_url_list:
            all_urls.append({
                "Video URL": video_url,
                "Original URL": split_url
            })

    # 不要なURLをフィルタリング
    filtered_all_urls = []
    for item in all_urls:
        corrected_urls = filter_and_correct_urls([item["Original URL"]])
        for corrected_url in corrected_urls:
            filtered_all_urls.append({
                "Video URL": item["Video URL"],
                "Original URL": corrected_url
            })

    download_records = []
    bar = progressbar.ProgressBar(max_value=len(filtered_all_urls))

    engine = create_engine(db_url)
    inspector = inspect(engine)

    for i, item in enumerate(filtered_all_urls):
        video_url = item["Video URL"]
        original_url = item["Original URL"]
        video_id = get_video_id_from_url(remove_query_params(original_url))

        # 動画情報からチャンネル情報を取得
        channel_data_list = []
        video_info = search_data.get_video_details(video_id)
        if video_info:
            channel_id = video_info.get('channel_id', '')
            channel_details = search_data.get_channel_details(channel_id)
            if channel_details:
                # channel_id テーブルに入れる用のdict
                record = {
                    'channel_id': channel_details['channel_id'],
                    'channel_title': channel_details['title'],
                    'channel_url': channel_details['channel_url'],
                    'subscriber_count': channel_details['subscriber_count'],
                    'channel_description': channel_details['description'],
                    'channel_type': 'Source',
                    'created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                }
                channel_data_list.append(record)

        else:
            channel_id = "unknown_channel"

        try:
            # 1) チャンネル情報を保存 (Channel_id テーブル)
            save_channel_to_sql(channel_data_list, db_url=db_url)

            # 2) 動画ID テーブルが既にあるかチェック
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

            # 3) チャットデータの取得
            df_chat = chat_download_data(original_url)
            if df_chat is None or df_chat.empty:
                # チャットが取得できなければスキップ
                download_records.append({
                    "Video URL": video_url,
                    "Original URL": original_url,
                    "DB Table": video_id,
                    "Video ID": video_id,
                    "Status": "No Chat Found"
                })
                bar.update(i + 1)
                continue

            # 4) スタンプ解析
            stamp_mapping = {}
            df_chat, updated_stamp_mapping, stamps_data_list = message_stamp2text(
                df_chat,
                stamp_mapping,
                channel_id=channel_id,  # スタンプ保存時に使う
            )

            # 5) チャットデータを動画IDテーブルに保存
            save_to_sql(df_chat, db_url=db_url, table_name=video_id, if_exists="append")

            # 6) スタンプ情報を "Stamp_data" テーブルに保存
            #    channel_id と stamp_code の組み合わせで管理
            save_stamps_to_sql(stamps_data_list, db_url=db_url)

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


# --------------------------------------------------------------
# 7) スクリプト実行例
# --------------------------------------------------------------
if __name__ == "__main__":
    # 例: CSVファイルに含まれるURLを一括で取り込み、SQLに保存
    csv_file = "../data/test_videos_processed.csv"
    result_df = list_original_urls(
        csv_file,
        delete_multiple=True,
        url_column="Original videoURL",
        db_url="sqlite:///chat_data.db",
    )
    result_df.to_csv("../data/download_results.csv", index=False, encoding="utf-8-sig")
