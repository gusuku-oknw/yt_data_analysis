from chat_downloader import ChatDownloader

# YouTubeライブのURL
url = "https://www.youtube.com/watch?v=-1FjQf5ipUA"

# チャットダウンローダーを初期化
chat = ChatDownloader()

# チャットメッセージを取得
messages = chat.get_chat(url, end_time='0:01:00')

# メッセージをループして処理
for message in messages:
    # 必要なデータを抽出
    author = message.get('author', {})
    message_text = message.get('message', 'No message text')
    timestamp = message.get('timestamp', 'No timestamp')

    # スタンプ画像を格納するリスト
    stamp_images = []
    badge_details = []

    # 1. スタンプ画像（emotes）
    if 'emotes' in message:
        for emote in message['emotes']:
            if 'images' in emote:
                stamp_images.extend([img.get('url') for img in emote['images'] if 'url' in img])

    # 2. バッジ情報
    if 'badges' in author:
        for badge in author['badges']:
            # バッジ詳細情報の収集
            badge_info = {
                'title': badge.get('title'),
                'id': badge.get('id'),
                'name': badge.get('name'),
                'version': badge.get('version'),
                'icon_name': badge.get('icon_name'),
                'icons': [icon.get('url') for icon in badge.get('icons', []) if 'url' in icon],
                'description': badge.get('description'),
                'alternative_title': badge.get('alternative_title'),
                'click_action': badge.get('click_action'),
                'click_url': badge.get('click_url'),
            }
            badge_details.append(badge_info)

    # メッセージを表示
    print(f"Timestamp: {timestamp}")
    print(f"Author: {author.get('name', 'Unknown')}")
    print(f"Message: {message_text}")
    print(f"Stamp Images: {stamp_images if stamp_images else 'No stamp images found.'}")
    print("Badge Details:")
    if badge_details:
        for i, badge in enumerate(badge_details, 1):
            print(f"  Badge {i}:")
            for key, value in badge.items():
                if value is not None:
                    print(f"    {key}: {value}")
                else:
                    print(f"    {key}: N/A")
    else:
        print("  No badges found.")
    print("-" * 40)
