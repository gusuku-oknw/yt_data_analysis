from chat_downloader import ChatDownloader
import pandas as pd


def download_chat_data(url, end_time='0:01:00'):
    """
    指定されたYouTube URLからチャットデータを取得し、DataFrameに変換する。

    Parameters:
        url (str): YouTube動画のURL。
        end_time (str): チャットデータ取得の終了時間 (例: '0:01:00')。

    Returns:
        pd.DataFrame: チャットデータ。
    """
    messages_data = []

    try:
        # チャットデータを取得
        chat = ChatDownloader().get_chat(url, end_time=end_time)

        # チャットメッセージをループして処理
        for message in chat:
            # 基本データの抽出
            time_in_seconds = message.get('time_in_seconds', 'N/A')
            message_text = message.get('message', 'N/A')
            amount = message.get('money', {}).get('amount', 'N/A')

            # 投稿者情報
            author = message.get('author', {})
            author_details = {
                "Author Name": author.get('name', 'N/A'),
                "Author ID": author.get('id', 'N/A'),
            }

            # バッジ情報の抽出
            badges = author.get('badges', [])
            badge_details = []
            for badge in badges:
                badge_details.append({
                    "title": badge.get('title', 'N/A'),
                    "id": badge.get('id', 'N/A'),
                    "name": badge.get('name', 'N/A'),
                    "version": badge.get('version', 'N/A'),
                    "icon_name": badge.get('icon_name', 'N/A'),
                    "icons": [icon.get('url') for icon in badge.get('icons', []) if 'url' in icon],
                    "description": badge.get('description', 'N/A'),
                    "alternative_title": badge.get('alternative_title', 'N/A'),
                    "click_action": badge.get('click_action', 'N/A'),
                    "click_url": badge.get('click_url', 'N/A'),
                })

            # スタンプ画像（emotes, sticker_images）をリストにまとめる
            stamp_images = []

            # emotes（テキストスタンプや絵文字など）
            if 'emotes' in message:
                for emote in message['emotes']:
                    if 'images' in emote:
                        # 複数の画像サイズがある場合、URLがリストで格納されていることがある
                        stamp_images.extend([img.get('url') for img in emote['images'] if 'url' in img])

            # sticker_images（YouTubeのスタンプなど）
            if 'sticker_images' in message:
                # 複数パターンの画像サイズがある場合はリスト
                stamp_images.extend([img.get('url') for img in message['sticker_images'] if 'url' in img])

            # データをリストに追加
            messages_data.append({
                "Time_in_seconds": time_in_seconds,
                "Message": message_text,
                "Amount": amount,
                **author_details,
                "Badge Details": badge_details,
                # セミコロンで区切って文字列化
                "Stamp Images": ";".join(stamp_images) if stamp_images else "No stamp images"
            })

    except Exception as e:
        print(f"Error during chat download: {e}")
        return None

    # DataFrameに変換して返す
    return pd.DataFrame(messages_data)


if __name__ == "__main__":
    # YouTubeライブのURL
    url = "https://www.youtube.com/watch?v=-1FjQf5ipUA"

    # チャットデータを取得
    chat_data = download_chat_data(url)
    if chat_data is not None:
        # 先頭5行を表示
        print(chat_data.head())

        # CSVに保存（文字化け防止にutf-8-sigが一般的）
        chat_data.to_csv("chat_data.csv", index=False, encoding="utf-8-sig")
        print("CSVファイルに保存しました: chat_data.csv")
