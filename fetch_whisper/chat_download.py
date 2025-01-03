import os
import re
import json
import progressbar
import pandas as pd
from datetime import datetime
from urllib.parse import urlparse, parse_qs
from dotenv import load_dotenv
from sqlalchemy import create_engine, inspect, text
from chat_downloader import ChatDownloader
from openai import OpenAI


# =============================================================================
# 1) URLや動画IDを扱うユーティリティクラス
# =============================================================================
class YTURLUtils:
    @staticmethod
    def split_urls(row, allow_channels=False):
        """
        複数URLを分割してリストとして返す。チャンネルURLは除外する/しないを指定可能。
        """
        url_pattern = re.compile(r'https?://[^\s,]+')
        urls = url_pattern.findall(row)
        return YTURLUtils.filter_and_correct_urls(urls, allow_channels=allow_channels)

    @staticmethod
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

    @staticmethod
    def get_video_id_from_url(url):
        """
        URLからYouTube動画IDを抽出する。
        """
        parsed_url = urlparse(url)
        if "youtube.com" in parsed_url.netloc:
            query_params = parse_qs(parsed_url.query)
            if "v" in query_params:
                return query_params["v"][0]
            # /live/ 形式
            if "/live/" in parsed_url.path:
                return parsed_url.path.split("/")[-1]
        if "youtu.be" in parsed_url.netloc:
            return parsed_url.path.strip("/")
        current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        return f"{current_time}_unknown_video_id"

    @staticmethod
    def remove_query_params(url):
        """
        URLから不要なクエリパラメータを削除する。ただし "v=" は保持。
        """
        parsed_url = urlparse(url)
        query_params = parse_qs(parsed_url.query)
        if "v" in query_params:
            base_url = f"{parsed_url.scheme}://{parsed_url.netloc}{parsed_url.path}"
            return f"{base_url}?v={query_params['v'][0]}"
        else:
            return f"{parsed_url.scheme}://{parsed_url.netloc}{parsed_url.path}"


# =============================================================================
# 2) OpenAI を使って画像のテキストを抽出するクラス
# =============================================================================
class ImageText:
    def __init__(self):
        load_dotenv()
        self.client = OpenAI(api_key=os.environ['OpenAIKey'])

    def image2text(self, image_url):
        """
        画像URLから文字＆感情を抽出する。
        """
        if not image_url.strip():
            return "None"

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
            return extracted
        except Exception as e:
            # エラー時は内容を表示し、"エラーが発生しました" 的に返す
            print(f"[image2text Error] {e}")
            return "None: Unknown"


# =============================================================================
# 3) チャットデータのダウンロードとスタンプ解析を行うクラス
# =============================================================================
class ChatDataProcessor:
    def __init__(self, db_handler):
        """
        db_handler: DBHandler のインスタンス
        """
        self.db_handler = db_handler
        self.image_text_extractor = ImageText()

    def download_chat_data(self, url):
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
            print(f"[download_chat_data Error] {e}")
            return None

        if not messages_data:
            return None

        df = pd.DataFrame(messages_data)
        return df

    def message_stamp2text(self, df, stamp_mapping, channel_id):
        """
        チャットのメッセージを解析し、スタンプコード・スタンプ説明などを抽出する。
        """
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

                # 1) 同じ行内 or stamp_mapping で既に判明していれば抑制（従来処理）
                if stamp in stamps_in_this_row:
                    reuse_text, reuse_emotion = stamps_in_this_row[stamp]
                    stamp_texts.append(f"{stamp_description}: {reuse_text}")
                    stamp_emotions.append(reuse_emotion)
                    continue

                if (stamp in stamp_mapping and
                        not stamp_mapping[stamp].startswith("Unknown Stamp:")):
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

                # 2) すでに (channel_id, stamp_code) が 5件以上あるならスキップ
                stamp_count_in_db = self.db_handler.count_stamp_occurrences(channel_id, stamp)
                if stamp_count_in_db >= 5:
                    # API 呼び出しをせず、固定値で処理
                    stamp_text = "Skipped (Limit Reached)"
                    stamp_emotion = "Unknown"
                    stamp_texts.append(f"{stamp_description}: {stamp_text}")
                    stamp_emotions.append(stamp_emotion)
                    stamps_in_this_row[stamp] = (stamp_text, stamp_emotion)

                    # DB にも追加しない → 「既に 5 つあるなら、これ以上詳細を増やさない」方針なら
                    # 何もしない or あえてレコード挿入したくない場合は追加しない
                    # ただし「何が起きたか履歴で見たい場合」には挿入してもよい
                    # stamps_data_list.append(...)
                    continue

                # 3) ここまで来たら新規に API  呼び出し
                if row['Stamp Image URL'] != "No stamp image":
                    extracted_text = self.image_text_extractor.image2text(row['Stamp Image URL'])
                    if ": " in extracted_text:
                        stamp_text, stamp_emotion = extracted_text.split(": ", 1)
                    else:
                        stamp_text, stamp_emotion = extracted_text, "Unknown"

                    # stamp_mapping を更新
                    if stamp_description.startswith("Unknown Stamp:"):
                        stamp_mapping[stamp] = f"{stamp_text}: {stamp_emotion}"

                    stamp_texts.append(f"{stamp_description}: {stamp_text}")
                    stamp_emotions.append(stamp_emotion)
                    stamps_in_this_row[stamp] = (stamp_text, stamp_emotion)

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

                    stamps_data_list.append({
                        "channel_id": channel_id,
                        "stamp_code": stamp,
                        "stamp_text": stamp_text,
                        "stamp_emotion": stamp_emotion,
                        "created_at": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    })

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


# =============================================================================
# 4) データベースへの保存（Channel・スタンプ・チャット）を扱うクラス
# =============================================================================
class DBHandler:
    def __init__(self, db_url="sqlite:///chat_data.db"):
        self.db_url = db_url
        self.engine = create_engine(db_url)

        self.create_stamp_data_table(self.engine)

    def create_stamp_data_table(self, engine):
        create_table_query = text("""
        CREATE TABLE IF NOT EXISTS Stamp_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            channel_id TEXT NOT NULL,
            stamp_code TEXT NOT NULL,
            stamp_text TEXT,
            stamp_emotion TEXT,
            created_at TEXT
        )
        """)
        with engine.connect() as conn:
            conn.execute(create_table_query)
        print("Stamp_data テーブルを作成しました（または既に存在します）。")

    def count_stamp_occurrences(self, channel_id, stamp_code):
        """
        Stamp_data テーブルで (channel_id, stamp_code) のレコードが何件あるかを返す
        """
        from sqlalchemy import text

        query = text("""
            SELECT COUNT(*) 
            FROM Stamp_data 
            WHERE channel_id = :ch_id AND stamp_code = :st_code
        """)

        with self.engine.connect() as conn:
            result = conn.execute(query, {"ch_id": channel_id, "st_code": stamp_code}).fetchone()

        if result is not None:
            count = result[0]  # 最初の要素がカウント値
            return count
        return 0

    def table_exists(self, table_name):
        inspector = inspect(self.engine)
        return table_name in inspector.get_table_names()

    def save_to_sql(self, dataframe, table_name, if_exists="append"):
        """
        汎用: DataFrame → SQL
        """
        dataframe.to_sql(table_name, con=self.engine, if_exists=if_exists, index=False)
        print(f"[SAVE] {table_name} に {len(dataframe)} 件のレコードを追加")

    def load_channel_record(self, channel_id):
        """
        channel_id をキーにして既存のレコードを DataFrame で取得
        見つからなければ空の DataFrame を返す
        """
        query = f"SELECT * FROM Channel_id WHERE channel_id = '{channel_id}'"
        try:
            df = pd.read_sql(query, self.engine)
            return df
        except:
            return pd.DataFrame()

    def upsert_channel_data(self, channel_data: dict, is_clipping=False, snippet_video_url=None, video_id=None):
        """
        channel_data は以下のキーを含む:
          {
            'channel_id': ...,
            'channel_title': ...,
            'channel_url': ...,
            'subscriber_count': ...,
            'channel_description': ...,
            'channel_type': 'Source' or 'clipping',
            'created_at': ...,
          }
        ここに 'video_ids' (JSON) や 'clipping_urls' (JSON) を追加管理する。
        すでにDBにレコードがある場合は、JSONカラムを取り出して追加する形。
        """
        channel_id = channel_data['channel_id']
        existing_df = self.load_channel_record(channel_id)

        # DBに該当レコードがない → 新規レコード作成
        if existing_df.empty:
            channel_data['video_ids'] = "[]"
            channel_data['clipping_urls'] = "[]"
            if is_clipping and snippet_video_url:
                # clipping
                channel_data['channel_type'] = 'clipping'
                channel_data['clipping_urls'] = json.dumps([snippet_video_url], ensure_ascii=False)
            elif video_id:
                # source
                channel_data['channel_type'] = 'Source'
                channel_data['video_ids'] = json.dumps([video_id], ensure_ascii=False)
            df_new = pd.DataFrame([channel_data])
            self.save_to_sql(df_new, "Channel_id", if_exists="append")
        else:
            # 既存レコード → JSON を読み込んで追加し再保存 (UPDATE)
            existing_record = existing_df.iloc[0].to_dict()

            video_ids_list = json.loads(existing_record.get('video_ids', '[]') or '[]')
            clipping_list = json.loads(existing_record.get('clipping_urls', '[]') or '[]')

            existing_type = existing_record.get('channel_type', 'Source')

            # クリッピングの情報を追記
            if is_clipping and snippet_video_url:
                existing_type = 'clipping'
                clipping_list.append(snippet_video_url)
                clipping_list = list(set(clipping_list))  # 重複排除
            elif video_id:  # ソース動画
                video_ids_list.append(video_id)
                video_ids_list = list(set(video_ids_list))

            updated_data = {
                'channel_id': existing_record['channel_id'],
                'channel_title': channel_data.get('channel_title', existing_record['channel_title']),
                'channel_url': channel_data.get('channel_url', existing_record['channel_url']),
                'subscriber_count': channel_data.get('subscriber_count', existing_record['subscriber_count']),
                'channel_description': channel_data.get('channel_description', existing_record['channel_description']),
                'channel_type': existing_type,
                'created_at': existing_record['created_at'],  # 初回記録日時を維持
                'video_ids': json.dumps(video_ids_list, ensure_ascii=False),
                'clipping_urls': json.dumps(clipping_list, ensure_ascii=False),
            }
            # 既存レコード削除して入れ直す
            delete_query = text(f"DELETE FROM Channel_id WHERE channel_id = '{channel_id}'")
            with self.engine.connect() as conn:
                conn.execute(delete_query)

            df_upd = pd.DataFrame([updated_data])
            self.save_to_sql(df_upd, "Channel_id", if_exists="append")

    def save_stamps_to_sql(self, stamps_data_list):
        """
        スタンプ情報を "Stamp_data" テーブルに保存する
        """
        if not stamps_data_list:
            return
        df_stamps = pd.DataFrame(stamps_data_list)
        self.save_to_sql(df_stamps, table_name="Stamp_data", if_exists="append")


# =============================================================================
# 5) メインの処理: CSV 読み込み → チャット/チャンネル/スタンプ保存
# =============================================================================
class ChatScraper:
    def __init__(self, db_url="sqlite:///chat_data.db"):
        self.db_handler = DBHandler(db_url=db_url)
        self.chat_processor = ChatDataProcessor(db_handler=self.db_handler)

        from search_yt import search_yt
        self.search_data = search_yt()

    def list_original_urls(self,
                           csv_file,
                           url_column="Original videoURL",
                           video_url_column="Video URL",
                           delete_multiple=False):
        """
        CSV のオリジナルURLからチャットを取得し、DBに保存する。
          - 動画チャット: テーブル名は "video_id" (YouTubeのID)
          - チャンネル情報: テーブル名 "Channel_id"
          - スタンプ情報: テーブル名 "Stamp_data"
          - "Video URL" は切り抜きチャンネルとして登録 ( channel_type='clipping' )
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

        # すべてのURLを集約
        all_urls = []
        for _, row in urls.iterrows():
            snippet_url = row[video_url_column]      # 切り抜き
            original_url = row[url_column]           # 本家配信
            split_url_list = YTURLUtils.split_urls(original_url)
            if delete_multiple and len(split_url_list) > 1:
                continue
            for split_url in split_url_list:
                all_urls.append({
                    "snippet_url": snippet_url,
                    "original_url": split_url
                })

        # フィルタリング
        filtered_all_urls = []
        for item in all_urls:
            corrected_urls = YTURLUtils.filter_and_correct_urls([item["original_url"]])
            for corrected_url in corrected_urls:
                filtered_all_urls.append({
                    "snippet_url": item["snippet_url"],
                    "original_url": corrected_url
                })

        download_records = []
        bar = progressbar.ProgressBar(max_value=len(filtered_all_urls))

        for i, item in enumerate(filtered_all_urls):
            snippet_url = item["snippet_url"]
            original_url = item["original_url"]

            # 切り抜き動画IDを取得
            snippet_url_clean = YTURLUtils.remove_query_params(snippet_url)
            snippet_video_id = YTURLUtils.get_video_id_from_url(snippet_url_clean)

            # オリジナル動画ID
            original_url_clean = YTURLUtils.remove_query_params(original_url)
            original_video_id = YTURLUtils.get_video_id_from_url(original_url_clean)

            # 切り抜きチャンネル
            snippet_info = self.search_data.get_video_details(snippet_video_id)
            snippet_channel_id = None
            if snippet_info:
                snippet_channel_id = snippet_info.get('channel_id', '')

            # オリジナル配信チャンネル
            original_info = self.search_data.get_video_details(original_video_id)
            channel_id_source = None
            if original_info:
                channel_id_source = original_info.get('channel_id', 'unknown_channel')

            try:
                # ------------------------------------------------------
                # [A] 切り抜きチャンネルを Channel_id に upsert
                # ------------------------------------------------------
                if snippet_channel_id:
                    snippet_channel_details = self.search_data.get_channel_details(snippet_channel_id)
                    if snippet_channel_details:
                        snippet_channel_data = {
                            'channel_id': snippet_channel_details['channel_id'],
                            'channel_title': snippet_channel_details['title'],
                            'channel_url': snippet_channel_details['channel_url'],
                            'subscriber_count': snippet_channel_details['subscriber_count'],
                            'channel_description': snippet_channel_details['description'],
                            'channel_type': 'clipping',  # クリッピング
                            'created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        }
                        # snippet_url を clipping_urls に追加
                        self.db_handler.upsert_channel_data(
                            snippet_channel_data,
                            is_clipping=True,
                            snippet_video_url=snippet_url
                        )

                # ------------------------------------------------------
                # [B] オリジナル配信チャンネルを Channel_id に upsert
                # ------------------------------------------------------
                if channel_id_source:
                    source_channel_details = self.search_data.get_channel_details(channel_id_source)
                    if source_channel_details:
                        source_channel_data = {
                            'channel_id': source_channel_details['channel_id'],
                            'channel_title': source_channel_details['title'],
                            'channel_url': source_channel_details['channel_url'],
                            'subscriber_count': source_channel_details['subscriber_count'],
                            'channel_description': source_channel_details['description'],
                            'channel_type': 'Source',
                            'created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        }
                        self.db_handler.upsert_channel_data(
                            source_channel_data,
                            is_clipping=False,
                            video_id=original_video_id
                        )
                else:
                    channel_id_source = "unknown_channel"

                # ------------------------------------------------------
                # [C] チャットテーブル (オリジナル動画ID) の存在チェック
                # ------------------------------------------------------
                if self.db_handler.table_exists(original_video_id):
                    print(f"既存テーブルが見つかりました。スキップします: {original_video_id}")
                    download_records.append({
                        "Original URL": original_url,
                        "Snippet URL": snippet_url,
                        "DB Table": original_video_id,
                        "Video ID": original_video_id,
                        "Status": "Skipped (Already Exists)"
                    })
                    bar.update(i + 1)
                    continue

                # ------------------------------------------------------
                # [D] チャットのダウンロード
                # ------------------------------------------------------
                df_chat = self.chat_processor.download_chat_data(original_url)
                if df_chat is None or df_chat.empty:
                    # チャットが取得できなければスキップ
                    download_records.append({
                        "Original URL": original_url,
                        "Snippet URL": snippet_url,
                        "DB Table": original_video_id,
                        "Video ID": original_video_id,
                        "Status": "No Chat Found"
                    })
                    bar.update(i + 1)
                    continue

                # ------------------------------------------------------
                # [E] スタンプ解析
                # ------------------------------------------------------
                stamp_mapping = {}
                df_chat, updated_stamp_mapping, stamps_data_list = self.chat_processor.message_stamp2text(
                    df_chat,
                    stamp_mapping,
                    channel_id=channel_id_source
                )

                # ------------------------------------------------------
                # [F] チャットを動画IDテーブルに保存
                # ------------------------------------------------------
                self.db_handler.save_to_sql(df_chat, table_name=original_video_id)

                # ------------------------------------------------------
                # [G] スタンプ情報を "Stamp_data" に保存
                # ------------------------------------------------------
                self.db_handler.save_stamps_to_sql(stamps_data_list)

                download_records.append({
                    "Original URL": original_url,
                    "Snippet URL": snippet_url,
                    "DB Table": original_video_id,
                    "Video ID": original_video_id,
                    "Status": "Downloaded"
                })

            except Exception as e:
                print(f"エラーが発生しました: {e} - URL: {original_url}")

            bar.update(i + 1)

        print("\nすべての処理が完了しました")
        return pd.DataFrame(download_records)


# =============================================================================
# 6) スクリプト実行例
# =============================================================================
if __name__ == "__main__":
    csv_file = "../data/にじさんじ　切り抜き_20250102_202807.csv"
    scraper = ChatScraper(db_url="sqlite:///chat_data.db")
    result_df = scraper.list_original_urls(
        csv_file,
        delete_multiple=True,
        url_column="Original videoURL",
        video_url_column="Video URL",
    )
    result_df.to_csv("../data/download_results.csv", index=False, encoding="utf-8-sig")
