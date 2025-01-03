from chat_downloader import ChatDownloader
import pandas as pd
import re  # 数値データ抽出用


def download_chat_data(url, end_time='0:01:00', output_file='chat_data_with_members.csv'):
    """
    指定されたYouTube URLからチャットデータを取得し、メンバー情報を数値データに変換し保存する。

    Parameters:
        url (str): YouTube動画のURL。
        end_time (str): チャットデータ取得の終了時間 (例: '0:01:00')。
        output_file (str): 保存するCSVファイルの名前。

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
            member_info = 0  # デフォルト値は0（非メンバー）
            badge_icon_url = ""
            for badge in badges:
                title = badge.get('title', 'N/A')
                icons = badge.get('icons', [])
                if icons:
                    # 最初のURLのみを保存
                    badge_icon_url = icons[0].get('url', '')

                # "Member"が含まれる場合のみメンバー情報を抽出し、数値に変換
                if "Member" in title:
                    match = re.search(r"(\d+)\s*(year|month)", title, re.IGNORECASE)
                    if match:
                        number = int(match.group(1))
                        unit = match.group(2).lower()
                        # 年数を月数に変換（必要なら）
                        if unit == "year":
                            member_info = number * 12
                        elif unit == "month":
                            member_info = number

            # スタンプ画像（emotes, sticker_images）の最初のURLを取得
            stamp_image_url = None
            if 'emotes' in message:
                for emote in message['emotes']:
                    if 'images' in emote:
                        # 最初の画像URLを取得
                        stamp_image_url = emote['images'][0].get('url', None)
                        break  # 最初のURLのみ取得して終了

            if not stamp_image_url and 'sticker_images' in message:
                # sticker_images から最初のURLを取得
                stamp_image_url = message['sticker_images'][0].get('url', None)

            # データをリストに追加
            messages_data.append({
                "Time_in_seconds": time_in_seconds,
                "Message": message_text,
                "Amount": amount,
                **author_details,
                "Member Info (Months)": member_info,  # メンバー情報（月数）
                "Badge Icon URL": badge_icon_url,  # 最初のアイコンURL
                "Stamp Image URL": stamp_image_url if stamp_image_url else "No stamp image"
            })

    except Exception as e:
        print(f"Error during chat download: {e}")
        return None

    # DataFrameに変換して返す
    df = pd.DataFrame(messages_data)

    # DataFrameをCSVに保存
    df.to_csv(output_file, index=False, encoding="utf-8-sig")
    print(f"チャットデータを '{output_file}' に保存しました。")
    return df


if __name__ == "__main__":
    # YouTubeライブのURL
    url = "https://www.youtube.com/watch?v=-1FjQf5ipUA"

    # チャットデータを取得して保存
    chat_data = download_chat_data(url)
    if chat_data is not None:
        print(chat_data.head())
