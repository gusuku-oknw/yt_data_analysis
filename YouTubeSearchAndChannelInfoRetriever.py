from googleapiclient.discovery import build
import os
from dotenv import load_dotenv
import csv
import re

# 環境変数の読み込み
load_dotenv()

# YouTube Data APIのAPIキーを設定
API_KEY = os.environ['YoutubeKey']

# YouTube Data APIのクライアントを作成
youtube_api = build('youtube', 'v3', developerKey=API_KEY)

# ISO 8601の再生時間を人間が読みやすい形式に変換する関数
def convert_duration(iso_duration):
    pattern = re.compile(r'PT(\d+H)?(\d+M)?(\d+S)?')
    matches = pattern.match(iso_duration)
    if not matches:
        return "N/A"
    hours = matches.group(1)[:-1] if matches.group(1) else 0
    minutes = matches.group(2)[:-1] if matches.group(2) else 0
    seconds = matches.group(3)[:-1] if matches.group(3) else 0
    hours = int(hours) if hours else 0
    minutes = int(minutes) if minutes else 0
    seconds = int(seconds) if seconds else 0
    if hours > 0:
        return f"{hours}:{minutes:02}:{seconds:02}"
    else:
        return f"{minutes}:{seconds:02}"

# 動画とチャンネル情報を取得するメソッド
def get_video_and_channel_data(video_id, channel_id):
    # 動画情報の取得
    video_response = youtube_api.videos().list(
        part="snippet,contentDetails,statistics",
        id=video_id
    ).execute()

    video_title = video_response['items'][0]['snippet']['title']
    video_description = video_response['items'][0]['snippet'].get('description', "N/A")
    iso_duration = video_response['items'][0]['contentDetails'].get('duration', "N/A")
    video_duration = convert_duration(iso_duration)
    thumbnails = video_response['items'][0]['snippet']['thumbnails']
    thumbnail_url = thumbnails.get('high', thumbnails.get('default', {})).get('url', "N/A")
    view_count = video_response['items'][0]['statistics'].get('viewCount', "N/A")

    # チャンネル情報の取得
    channel_response = youtube_api.channels().list(
        part="snippet,statistics",
        id=channel_id
    ).execute()

    channel_title = channel_response['items'][0]['snippet']['title']
    channel_description = channel_response['items'][0]['snippet'].get('description', "N/A")
    subscriber_count = channel_response['items'][0]['statistics'].get('subscriberCount', "非公開")

    # データを辞書形式で返す
    return {
        "Video Title": video_title,
        "Video URL": f"https://www.youtube.com/watch?v={video_id}",
        "Thumbnail URL": thumbnail_url,
        "Video Description": video_description,
        "Video Duration": video_duration,
        "Video Views": view_count,
        "Channel Title": channel_title,
        "Channel URL": f"https://www.youtube.com/channel/{channel_id}",
        "Subscriber Count": subscriber_count,
        "Channel Description": channel_description
    }

# 検索キーワードを指定
search_keyword = "Python tutorial"

# YouTube Data APIで検索を実行
response = youtube_api.search().list(
    q=search_keyword,
    part="snippet",
    type="video",
    maxResults=10
).execute()

# データを保存するリスト
data = []

# 検索結果の処理
for item in response['items']:
    video_id = item['id']['videoId']
    channel_id = item['snippet']['channelId']

    # 動画とチャンネルの情報を取得してリストに追加
    data.append(get_video_and_channel_data(video_id, channel_id))

# CSVファイルの名前
csv_filename = "youtube_search_results_with_channel.csv"

# CSVファイルにデータを書き込む
with open(csv_filename, mode="w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(file, fieldnames=[
        "Video Title", "Video URL", "Thumbnail URL", "Video Description",
        "Video Duration", "Video Views", "Channel Title", "Channel URL",
        "Subscriber Count", "Channel Description"
    ])
    writer.writeheader()
    writer.writerows(data)

print(f"データが {csv_filename} に保存されました。")
