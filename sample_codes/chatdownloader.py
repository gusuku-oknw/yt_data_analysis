from chat_downloader import ChatDownloader

# YouTubeライブのURL
url = "https://www.youtube.com/watch?v=-1FjQf5ipUA"

# チャットダウンローダーを初期化
chat = ChatDownloader()

# チャットメッセージを取得
messages = chat.get_chat(url)

# メッセージをループして処理
for message in messages:
    # 必要なデータを抽出
    author = message.get('author', {})
    message_text = message.get('message', 'No message text')
    timestamp = message.get('timestamp', 'No timestamp')

    # 画像URLを格納するリスト
    all_images = []

    # 1. スタンプ画像（emotes）
    if 'emotes' in message:
        for emote in message['emotes']:
            if 'images' in emote:
                all_images.extend([img.get('url') for img in emote['images'] if 'url' in img])

    # 2. 著者画像
    if 'images' in author:
        all_images.extend([img.get('url') for img in author['images'] if 'url' in img])

    # 3. バッジ画像
    if 'badges' in author:
        for badge in author['badges']:
            if 'icons' in badge:
                all_images.extend([icon.get('url') for icon in badge['icons'] if 'url' in icon])

    # メッセージを表示
    print(f"Timestamp: {timestamp}")
    print(f"Author: {author.get('name', 'Unknown')}")
    print(f"Message: {message_text}")
    if all_images:
        print(f"Images: {all_images}")
    else:
        print("No images found.")
    print("-" * 40)