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
from yt_url_utils import YTURLUtils

# =============================================================================
# 2) OpenAI を使って画像のテキストを抽出するクラス
# =============================================================================
class ImageText:
    def __init__(self):
        load_dotenv()
        self.client = OpenAI(api_key=os.environ['OpenAIKey'])

    def image2text(self, image_url):
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
            print(f"[image2text Error] {e}")
            return "None: Unknown"


# =============================================================================
# 3) チャットデータのダウンロードとスタンプ解析を行うクラス
# =============================================================================
class ChatDataProcessor:
    def __init__(self, db_handler):
        self.db_handler = db_handler
        self.image_text_extractor = ImageText()

    def download_chat_data(self, url):
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

                # 1) 同じ行内 or stamp_mapping で既に判明していれば抑制
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
                    stamp_text = "Skipped (Limit Reached)"
                    stamp_emotion = "Unknown"
                    stamp_texts.append(f"{stamp_description}: {stamp_text}")
                    stamp_emotions.append(stamp_emotion)
                    stamps_in_this_row[stamp] = (stamp_text, stamp_emotion)
                    continue

                # 3) 新規に API 呼び出し
                if row['Stamp Image URL'] != "No stamp image":
                    extracted_text = self.image_text_extractor.image2text(row['Stamp Image URL'])
                    if ": " in extracted_text:
                        stamp_text, stamp_emotion = extracted_text.split(": ", 1)
                    else:
                        stamp_text, stamp_emotion = extracted_text, "Unknown"

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
                json.dumps(stamps, ensure_ascii=False),
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
    def __init__(self, db_url=None):
        # -- 絶対パスで指定する例 ------------------------------------------------
        base_dir = os.path.dirname(os.path.abspath(__file__))  # スクリプトファイルのあるパス
        if db_url is None:
            db_path = os.path.join(base_dir, "chat_data.db")
            db_url = f"sqlite:///{db_path}"
        self.db_url = db_url
        # echo=True で実際のSQLをログ出力すると原因追跡がしやすくなる
        self.engine = create_engine(self.db_url, echo=False)

        # テーブルの作成
        self.create_channel_id_table()
        self.create_stamp_data_table()
        self.create_channel_videos_table()
        self.create_emotion_analysis_table()
        self.create_segment_comparisons_table()

    def create_channel_id_table(self):
        """
        Channel_id テーブルが無い場合に作成
        """
        create_table_query = text("""
        CREATE TABLE IF NOT EXISTS Channel_id (
            channel_id TEXT PRIMARY KEY,
            channel_title TEXT,
            channel_url TEXT,
            subscriber_count INTEGER,
            channel_description TEXT,
            channel_type TEXT,
            created_at TEXT
        )
        """)
        with self.engine.begin() as conn:
            conn.execute(create_table_query)
        print("Channel_id テーブルを作成しました（または既に存在します）。")

    def create_stamp_data_table(self):
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
        with self.engine.begin() as conn:
            conn.execute(create_table_query)
        print("Stamp_data テーブルを作成しました（または既に存在します）。")

    def create_channel_videos_table(self):
        create_table_query = text("""
        CREATE TABLE IF NOT EXISTS Channel_videos (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            channel_id TEXT NOT NULL,
            video_id TEXT NOT NULL,
            video_url TEXT NOT NULL,
            video_type TEXT NOT NULL,
            file_path TEXT,
            created_at TEXT
        )
        """)
        with self.engine.begin() as conn:
            conn.execute(create_table_query)
        print("Channel_videos テーブルを作成しました（または既に存在します）。")

    def create_emotion_analysis_table(self):
        create_table_query = text("""
        CREATE TABLE IF NOT EXISTS Emotion_analysis (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            video_id TEXT NOT NULL,
            analysis_method TEXT NOT NULL,
            sentiment REAL,
            weime REAL,
            mlask REAL,
            created_at TEXT
        )
        """)
        with self.engine.begin() as conn:
            conn.execute(create_table_query)
        print("Emotion_analysis テーブルを作成しました（または既に存在します）。")

    def create_segment_comparisons_table(self):
        create_table_query = text("""
        CREATE TABLE IF NOT EXISTS Segment_comparisons (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source_video_id TEXT NOT NULL,
            clip_video_id TEXT NOT NULL,
            match_text TEXT,
            source_text TEXT,
            clip_text TEXT,
            similarity REAL,
            created_at TEXT
        )
        """)
        with self.engine.begin() as conn:
            conn.execute(create_table_query)
        print("Segment_comparisons テーブルを作成しました（または既に存在します）。")

    def count_stamp_occurrences(self, channel_id, stamp_code):
        query = text("""
            SELECT COUNT(*)
            FROM Stamp_data
            WHERE channel_id = :ch_id AND stamp_code = :st_code
        """)
        with self.engine.begin() as conn:
            result = conn.execute(query, {"ch_id": channel_id, "st_code": stamp_code}).fetchone()
        if result is not None:
            return result[0]
        return 0

    def table_exists(self, table_name):
        inspector = inspect(self.engine)
        return table_name in inspector.get_table_names()

    # -------------------------------------------------------------------------
    # レコード単位で Insert or Update を行うメソッド
    # -------------------------------------------------------------------------
    def insert_or_update_rows(self, df: pd.DataFrame, table_name: str, primary_keys: list):
        if df is None or df.empty:
            return

        columns = df.columns.tolist()
        # UPDATE用のSET句: primary_keys 以外の列を更新対象
        set_clause = ", ".join([f"{col} = :{col}" for col in columns if col not in primary_keys])
        where_clause = " AND ".join([f"{pk} = :{pk}" for pk in primary_keys])

        insert_cols_str = ", ".join(columns)
        insert_vals_str = ", ".join([f":{col}" for col in columns])

        with self.engine.begin() as conn:  # begin() により自動コミット
            for row in df.to_dict(orient="records"):
                # まず既存レコードがあるかチェック
                select_query = text(f"SELECT COUNT(*) FROM {table_name} WHERE {where_clause}")
                select_params = {pk: row[pk] for pk in primary_keys}
                record_exists = conn.execute(select_query, select_params).fetchone()[0] > 0

                if record_exists:
                    if set_clause.strip():
                        update_query = text(f"""
                            UPDATE {table_name}
                            SET {set_clause}
                            WHERE {where_clause}
                        """)
                        conn.execute(update_query, row)
                else:
                    insert_query = text(f"""
                        INSERT INTO {table_name} ({insert_cols_str})
                        VALUES ({insert_vals_str})
                    """)
                    conn.execute(insert_query, row)

    def save_to_sql(self, dataframe, table_name, primary_keys=None):
        if dataframe is None or dataframe.empty:
            print(f"[SAVE] {table_name} に保存するデータがありません。")
            return

        # primary_keys が無い場合はそのまま append（従来動作）
        if not primary_keys:
            # replace でなく append に注意
            dataframe.to_sql(table_name, con=self.engine, if_exists="append", index=False)
            print(f"[SAVE - append only] {table_name} に {len(dataframe)} 件のレコードを追加")
            return

        # テーブル未作成なら新規作成
        if not self.table_exists(table_name):
            with self.engine.begin() as conn:
                dataframe.to_sql(table_name, con=conn, if_exists="replace", index=False)
            print(f"[SAVE - created new table] {table_name} に {len(dataframe)} 件のレコードを新規挿入")
            return

        # テーブルが既に存在 → insert or update
        self.insert_or_update_rows(dataframe, table_name, primary_keys=primary_keys)
        print(f"[SAVE] {table_name} に {len(dataframe)} 件のレコードをInsert/Updateしました。")

    def load_channel_record(self, channel_id):
        query = f"SELECT * FROM Channel_id WHERE channel_id = '{channel_id}'"
        with self.engine.begin() as conn:
            try:
                df = pd.read_sql(query, conn)
                return df
            except:
                return pd.DataFrame()

    # -------------------------------------------------------------------------
    # Channel_id テーブルへのアップサート
    # -------------------------------------------------------------------------
    def upsert_channel_data(self, channel_data: dict):
        """
        channel_data は以下のキーを含む想定:
          {
            'channel_id': ...,
            'channel_title': ...,
            'channel_url': ...,
            'subscriber_count': ...,
            'channel_description': ...,
            'channel_type': 'Source' or 'clipping',
            'created_at': ...
          }
        """
        channel_id = channel_data['channel_id']
        existing_df = self.load_channel_record(channel_id)

        if existing_df.empty:
            # 新規 INSERT
            df_new = pd.DataFrame([channel_data])
            self.save_to_sql(df_new, "Channel_id", primary_keys=["channel_id"])
        else:
            # UPDATE
            existing_record = existing_df.iloc[0].to_dict()
            updated_data = {
                'channel_id': channel_id,
                'channel_title': channel_data.get('channel_title', existing_record.get('channel_title')),
                'channel_url': channel_data.get('channel_url', existing_record.get('channel_url')),
                'subscriber_count': channel_data.get('subscriber_count', existing_record.get('subscriber_count')),
                'channel_description': channel_data.get('channel_description', existing_record.get('channel_description')),
                'channel_type': channel_data.get('channel_type', existing_record.get('channel_type')),
                'created_at': existing_record.get('created_at', channel_data.get('created_at')),
            }
            df_upd = pd.DataFrame([updated_data])
            self.save_to_sql(df_upd, "Channel_id", primary_keys=["channel_id"])

    # -------------------------------------------------------------------------
    # Channel_videos テーブル への動画情報アップサート
    # -------------------------------------------------------------------------
    def upsert_channel_videos(self, channel_id, video_id, video_url, video_type, file_path=None):
        """
        Channel_videos テーブルに (channel_id, video_id, video_url, video_type, file_path) を登録
        """
        # 既にあるかどうか確認
        select_query = text("""
            SELECT COUNT(*) FROM Channel_videos
            WHERE channel_id = :ch_id AND video_id = :v_id
        """)
        with self.engine.begin() as conn:
            result = conn.execute(select_query, {"ch_id": channel_id, "v_id": video_id}).fetchone()
            exists = (result[0] > 0)

        if exists:
            print(f"[Channel_videos] 既に登録済み: {channel_id} - {video_id}")
            return

        now_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        insert_query = text("""
            INSERT INTO Channel_videos (channel_id, video_id, video_url, video_type, file_path, created_at)
            VALUES (:ch_id, :v_id, :v_url, :v_type, :f_path, :created_at)
        """)
        with self.engine.begin() as conn:
            conn.execute(insert_query, {
                "ch_id": channel_id,
                "v_id": video_id,
                "v_url": video_url,
                "v_type": video_type,
                "f_path": file_path,
                "created_at": now_str
            })
        print(f"[Channel_videos] 新規登録: {channel_id} - {video_id}")

    def save_stamps_to_sql(self, stamps_data_list):
        if not stamps_data_list:
            return
        df_stamps = pd.DataFrame(stamps_data_list)
        self.save_to_sql(
            df_stamps,
            table_name="Stamp_data",
            primary_keys=["channel_id", "stamp_code", "stamp_text", "stamp_emotion", "created_at"]
        )

    # -------------------------------------------------------------------------
    # Emotion_analysis テーブルへの保存
    # -------------------------------------------------------------------------
    def save_emotion_analysis(self, video_id, analysis_results):
        """
        analysis_results は以下の形式を想定:
        {
            "sentiment": float,
            "weime": float,
            "mlask": float
        }
        """
        now_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        data = {
            "video_id": video_id,
            "analysis_method": "default",  # 必要に応じて変更
            "sentiment": analysis_results.get("sentiment"),
            "weime": analysis_results.get("weime"),
            "mlask": analysis_results.get("mlask"),
            "created_at": now_str
        }
        df = pd.DataFrame([data])
        self.save_to_sql(df, "Emotion_analysis", primary_keys=["video_id", "analysis_method"])
        print(f"[Emotion_analysis] 保存しました: {video_id}")

    # -------------------------------------------------------------------------
    # Segment_comparisons テーブルへの保存
    # -------------------------------------------------------------------------
    def save_segment_comparisons(self, source_video_id, clip_video_id, matches):
        """
        matches は以下の形式を想定:
        [
            {
                "clip_text": "切り抜きテキスト" または None,
                "clip_start": float または None,
                "clip_end": float または None,
                "source_text": "元テキスト" または None,
                "source_start": float または None,
                "source_end": float または None,
                "similarity": float または None
            },
            ...
        ]
        """
        now_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        data = []
        for match in matches:
            # 一致していない場合の処理
            match_text = None
            if match.get("clip_text") and match.get("source_text"):
                match_text = f"{match['clip_text']} => {match['source_text']}"

            record = {
                "source_video_id": source_video_id,
                "clip_video_id": clip_video_id,
                "match_text": match_text,
                "source_text": match.get("source_text"),
                "clip_text": match.get("clip_text"),
                "similarity": match.get("similarity"),
                "created_at": now_str
            }
            data.append(record)

        if data:
            df = pd.DataFrame(data)
            self.save_to_sql(df, "Segment_comparisons", primary_keys=["source_video_id", "clip_video_id", "match_text"])
            print(f"[Segment_comparisons] {len(data)} レコード保存しました。")
        else:
            print("[Segment_comparisons] 保存するデータがありません。")

    # 動画IDテーブルを取得する関数
    def fetch_table_from_db(self, video_id, db_handler):
        """
        動画IDに対応するテーブルを取得します。
        """
        table_name = f"{video_id}"
        try:
            # テーブルの存在確認
            query_check = f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table_name}'"
            result = pd.read_sql(query_check, db_handler.engine)
            if result.empty:
                print(f"テーブルが存在しません: {table_name}")
                return None

            # テーブルデータの取得
            query_fetch = f"SELECT * FROM {table_name}"
            table_data = pd.read_sql(query_fetch, db_handler.engine)
            return table_data
        except Exception as e:
            print(f"DB取得エラー: {e}")
            return None


# =============================================================================
# 5) メインの処理: CSV 読み込み → チャット/チャンネル/スタンプ保存
# =============================================================================
class ChatScraper:
    def __init__(self, db_url=None):
        self.db_handler = DBHandler(db_url=db_url)
        self.chat_processor = ChatDataProcessor(db_handler=self.db_handler)

        # ダミー: 実際には search_yt モジュールなどを import
        from search_yt import search_yt
        self.search_data = search_yt()

    def list_original_urls(self,
                           df: pd.DataFrame,
                           url_column="Original videoURL",
                           video_url_column="Video URL",
                           delete_multiple=False):
        try:
            if url_column not in df.columns and "Original videoURL" in df.columns:
                url_column = "Original videoURL"
            if url_column not in df.columns or video_url_column not in df.columns:
                print(f"指定されたカラム '{url_column}' または '{video_url_column}' がCSVファイルに存在しません。")
                return pd.DataFrame()
            urls = df[[video_url_column, url_column]].dropna()

        except Exception as e:
            print(f"エラーが発生しました: {str(e)}")
            return pd.DataFrame()

        # すべてのURLを集約
        all_urls = []
        for _, row in urls.iterrows():
            snippet_url = row[video_url_column]
            original_url = row[url_column]
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

            snippet_url_clean = YTURLUtils.remove_query_params(snippet_url)
            snippet_video_id = YTURLUtils.get_video_id_from_url(snippet_url_clean)

            original_url_clean = YTURLUtils.remove_query_params(original_url)
            original_video_id = YTURLUtils.get_video_id_from_url(original_url_clean)

            snippet_info = self.search_data.get_video_details(snippet_video_id)
            snippet_channel_id = None
            if snippet_info:
                snippet_channel_id = snippet_info.get('channel_id', '')

            original_info = self.search_data.get_video_details(original_video_id)
            channel_id_source = None
            if original_info:
                channel_id_source = original_info.get('channel_id', 'unknown_channel')

            try:
                # ------------------------------------------------------
                # [A] 切り抜きチャンネル
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
                            'channel_type': 'clipping',
                            'created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        }
                        self.db_handler.upsert_channel_data(snippet_channel_data)
                        self.db_handler.upsert_channel_videos(
                            channel_id=snippet_channel_details['channel_id'],
                            video_id=snippet_video_id,
                            video_url=snippet_url,
                            video_type='clipping'
                        )

                # ------------------------------------------------------
                # [B] オリジナル配信チャンネル
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
                        self.db_handler.upsert_channel_data(source_channel_data)
                        self.db_handler.upsert_channel_videos(
                            channel_id=source_channel_details['channel_id'],
                            video_id=original_video_id,
                            video_url=original_url,
                            video_type='Source'
                        )
                else:
                    channel_id_source = "unknown_channel"

                # ------------------------------------------------------
                # [C] すでにチャットテーブルが存在するかチェック
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
                # ※ 主キー候補: Time_in_seconds + Author ID + Message
                self.db_handler.save_to_sql(
                    df_chat,
                    table_name=original_video_id,
                    primary_keys=["Time_in_seconds", "Author ID", "Message"]
                )

                # ------------------------------------------------------
                # [G] スタンプ情報を Stamp_data に保存
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
    try:
        os.path.isfile(csv_file)
    except FileNotFoundError:
        print(f"CSVファイル '{csv_file}' が見つかりません。")
        exit()

    # Windowsパスの場合は「sqlite:///C:/～」の形式にする
    db_url = "sqlite:///C:/Users/tmkjn/Documents/python/data_analysis/fetch_whisper/chat_data.db"

    scraper = ChatScraper(db_url=db_url)

    result_df = scraper.list_original_urls(
        pd.read_csv(csv_file, encoding="utf-8-sig"),
        delete_multiple=True,
        url_column="Original videoURL",
        video_url_column="Video URL",
    )

    if not result_df.empty:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        out_csv_path = os.path.join(base_dir, "../data/download_results.csv")
        result_df.to_csv(out_csv_path, index=False, encoding="utf-8-sig")
        print(f"ダウンロード結果を {out_csv_path} に保存しました。")
    else:
        print("ダウンロード結果はありません。")
