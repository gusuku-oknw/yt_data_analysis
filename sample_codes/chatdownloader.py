from chat_downloader import ChatDownloader
import pandas as pd
import re  # 数値データ抽出用
import os
from dotenv import load_dotenv
from openai import OpenAI


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
            # 例: 'gpt-4o-mini' はダミーのモデル名。実際に利用するモデル名を指定してください。
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


def message_stamp2text(df, stamp_mapping):
    """
    DataFrame内のメッセージとスタンプ画像を処理し、新しいカラムを追加しながら
    stamp_mapping を image2text の結果で更新して返す。

    - 同じ行で同じスタンプコードが重複した場合、APIは1回のみ呼ばれる。
    - 複数行においても、一度APIで取得済みのスタンプコードがあれば再度APIを呼ばない。

    Parameters:
        df (pd.DataFrame): 元のチャットデータを含むDataFrame。
        stamp_mapping (dict): スタンプ種類と説明のマッピング辞書。

    Returns:
        (pd.DataFrame, dict): 新しいカラムを追加したDataFrameと更新されたstamp_mapping。
    """
    # ImageText のインスタンスを作成
    image_text_extractor = ImageText()

    def process_row(row):
        original_message = row['Message']
        message = original_message.replace('□', '').strip()  # '□'を削除
        stamps = []
        remaining_message = message

        # スタンプコード(':_xxx:')の抽出と削除
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

        # スタンプの種類と感情を保存
        stamp_texts = []
        stamp_emotions = []

        # 同一行内で同じ stamp_code が出てきたら、同じ結果を使い回すための辞書
        # 例: {"stamp_code1": ("テキスト", "感情"), "stamp_code2": ("テキスト", "感情")}
        stamps_in_this_row = {}

        for stamp in stamps:
            # 既存のスタンプ説明を取得 (まだ確定されていない場合は "Unknown Stamp: {stamp}" かも)
            stamp_description = stamp_mapping.get(stamp, f"Unknown Stamp: {stamp}")

            # すでに同じ行でAPIを呼んでいたら使い回す
            if stamp in stamps_in_this_row:
                reuse_text, reuse_emotion = stamps_in_this_row[stamp]
                stamp_texts.append(f"{stamp_description}: {reuse_text}")
                stamp_emotions.append(reuse_emotion)
                continue

            # もし stamp_mapping[stamp] が "Unknown Stamp: ..." でなければ
            # -> すでに過去行で抽出済み (かつ確定済み) とみなし、APIを呼ばない
            if stamp in stamp_mapping and not stamp_mapping[stamp].startswith("Unknown Stamp:"):
                # 既に確定情報あり => "text: emotion" の形で保存されていると想定
                known_text_emotion = stamp_mapping[stamp]
                # 例: "Hello: Joy"
                if ": " in known_text_emotion:
                    known_text, known_emotion = known_text_emotion.split(": ", 1)
                else:
                    # フォーマット崩れ対策
                    known_text, known_emotion = known_text_emotion, "Unknown"

                # 同じ行でも使い回せるように記録しておく
                stamps_in_this_row[stamp] = (known_text, known_emotion)
                stamp_texts.append(f"{stamp_description}: {known_text}")
                stamp_emotions.append(known_emotion)
                continue

            # ここまできたらまだ確定していない => APIを呼んで取得する
            if row['Stamp Image URL'] != "No stamp image":
                extracted_text = image_text_extractor.image2text(row['Stamp Image URL'])

                if ": " in extracted_text:
                    stamp_text, stamp_emotion = extracted_text.split(": ", 1)
                else:
                    stamp_text, stamp_emotion = extracted_text, "Unknown"

                # stamp_description が "Unknown Stamp:" のままだったら更新
                # あるいは stamp_mapping にキー自体なければ登録
                if stamp_description.startswith("Unknown Stamp:"):
                    stamp_mapping[stamp] = f"{stamp_text}: {stamp_emotion}"

                stamp_texts.append(f"{stamp_description}: {stamp_text}")
                stamp_emotions.append(stamp_emotion)

                # 同じ行内で次回以降使い回すため登録
                stamps_in_this_row[stamp] = (stamp_text, stamp_emotion)

            else:
                # 画像がない場合はそのまま Unknown
                stamp_text = "No image available"
                stamp_emotion = "Unknown"
                stamp_texts.append(f"{stamp_description}: {stamp_text}")
                stamp_emotions.append(stamp_emotion)

                # 同じ行内で使い回すため登録
                stamps_in_this_row[stamp] = (stamp_text, stamp_emotion)

        processed_message = remaining_message.strip()

        return (
            original_message,
            processed_message,
            stamps,
            "; ".join(stamp_texts),
            "; ".join(stamp_emotions),
        )

    # 各行を処理して新しいカラムを付与
    df[[
        'Original Message',
        'Message',
        'Stamp Codes',
        'Stamp Texts',
        'Stamp Emotions'
    ]] = df.apply(lambda row: pd.Series(process_row(row)), axis=1)

    return df, stamp_mapping


def download_chat_data(url, end_time='0:00:10'):
    """
    指定されたYouTube URLからチャットデータを取得し、メンバー情報を数値データに変換しDataFrameを返す。

    Parameters:
        url (str): YouTube動画のURL。
        end_time (str): チャットデータ取得の終了時間 (例: '0:01:00')。

    Returns:
        pd.DataFrame: チャットデータ。
    """
    messages_data = []

    try:
        # ChatDownloader を使ってチャットデータを取得
        chat = ChatDownloader().get_chat(url, end_time=end_time)

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

    df = pd.DataFrame(messages_data)
    return df


if __name__ == "__main__":
    # YouTubeライブのURL
    url = "https://www.youtube.com/watch?v=-1FjQf5ipUA"

    # チャットデータを取得
    chat_data = download_chat_data(url)

    # 既存マッピング（スタンプコードと説明の対応表）
    stamp_mapping = {
        # 既知のスタンプがあればここに書く
        # 例: '123': 'ハートのスタンプ'
    }

    if chat_data is not None:
        # チャットデータにスタンプの文字・感情を付与し、stamp_mapping も更新
        df, updated_stamp_mapping = message_stamp2text(
            chat_data,
            stamp_mapping,
        )

        # CSVとして保存
        df.to_csv("output.csv", index=False, encoding="utf-8-sig")

        # 結果の確認
        print(df.head())
        print("----- Updated Stamp Mapping -----")
        print(updated_stamp_mapping)
        print("処理結果を output.csv に保存しました。")
