from googleapiclient.discovery import build
from pytube import Channel
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

# チャンネル情報を取得するメソッド
def get_channel_videos_and_details(channel_id, max_results=10):
    channel_response = youtube_api.channels().list(
        part="snippet,statistics",
        id=channel_id
    ).execute()

    # チャンネル情報の取得
    channel_info = channel_response['items'][0]
    channel_title = channel_info['snippet']['title']
    channel_description = channel_info['snippet'].get('description', "N/A")
    subscriber_count = channel_info['statistics'].get('subscriberCount', "非公開")
    channel_url = f"https://www.youtube.com/channel/{channel_id}"

    # チャンネルの動画情報を取得
    search_response = youtube_api.search().list(
        part="snippet",
        channelId=channel_id,
        maxResults=max_results,
        order="date",  # 新着順
        type="video"
    ).execute()

    videos = []
    for item in search_response['items']:
        video_id = item['id']['videoId']
        video_title = item['snippet']['title']
        video_url = f"https://www.youtube.com/watch?v={video_id}"

        # 動画詳細情報の取得
        video_response = youtube_api.videos().list(
            part="snippet,contentDetails,statistics",
            id=video_id
        ).execute()

        video_info = video_response['items'][0]
        video_description = video_info['snippet'].get('description', "N/A")
        video_duration = convert_duration(video_info['contentDetails'].get('duration', "N/A"))
        view_count = video_info['statistics'].get('viewCount', "N/A")
        thumbnails = video_info['snippet']['thumbnails']
        thumbnail_url = thumbnails.get('high', thumbnails.get('default', {})).get('url', "N/A")

        # 動画データを追加
        videos.append({
            "Video Title": video_title,
            "Video URL": video_url,
            "Thumbnail URL": thumbnail_url,
            "Video Description": video_description,
            "Video Duration": video_duration,
            "Video Views": view_count
        })

    return {
        "Channel Title": channel_title,
        "Channel URL": channel_url,
        "Subscriber Count": subscriber_count,
        "Channel Description": channel_description,
        "Videos": videos
    }

# pytubeを使用して人気動画を取得するメソッド
def get_popular_videos_from_channel(channel_url, max_results=10):
    channel = Channel(channel_url)
    popular_videos = []

    for video in channel.videos[:max_results]:
        popular_videos.append({
            "Video Title": video.title,
            "Video URL": video.watch_url,
            "Views": video.views,
            "Published Date": video.publish_date
        })

    return popular_videos

# チャンネルIDまたはURLでテスト実行
channel_id = "UC8butISFwT-Wl7EV0hUK0BQ"  # example: freeCodeCamp.org
channel_details = get_channel_videos_and_details(channel_id)

# チャンネル人気動画の取得
channel_url = f"https://www.youtube.com/channel/{channel_id}"
popular_videos = get_popular_videos_from_channel(channel_url)

# 結果表示
print(f"Channel: {channel_details['Channel Title']} ({channel_details['Subscriber Count']} subscribers)")
print("Latest Videos:")
for video in channel_details["Videos"]:
    print(f"- {video['Video Title']} ({video['Video Views']} views)")

print("\nPopular Videos:")
for video in popular_videos:
    print(f"- {video['Video Title']} ({video['Views']} views)")
