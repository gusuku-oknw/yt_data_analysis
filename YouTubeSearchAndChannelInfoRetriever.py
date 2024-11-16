from datetime import datetime
import os
import re
import csv
from dotenv import load_dotenv
from googleapiclient.discovery import build
import pandas as pd
from openai import OpenAI

# 環境変数の読み込み
load_dotenv()
API_KEY = os.environ['YoutubeKey']
OpenAIKey = os.environ['OpenAIKey']
# YouTube Data APIのクライアントを作成
youtube = build('youtube', 'v3', developerKey=API_KEY)

client = OpenAI(
    api_key=OpenAIKey,
)

# テキストを入力として、OpenAIのGPT-4を使用してテキストを生成する関数
def prompt_clip_api_call(text):
    if not text.strip():
        return ""
    elif text == ",":
        return ""

    message = [
        {"role": "user", "content": "今から入力されるテキストは切り抜き動画の概要欄です。 \
        このテキストの中で元動画を指しているURLを教えて下さい。このときの返答としてはURLのみを出力してください \
        URLがない場合はNoneと返してください"},
        {"role": "user", "content": text}
    ]
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=message,
            temperature=0.2,
            # max_tokens=50
        )

        translated_text = response.choices[0].message.content.strip()
    except Exception as e:
        print("エラーが発生しました:", str(e))
        translated_text = "エラーが発生しました。"

    return translated_text


def convert_duration(iso_duration):
    """
    ISO 8601の再生時間をHH:MM:SS形式に変換する関数。
    """
    pattern = re.compile(r'PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?')
    matches = pattern.match(iso_duration)
    if not matches:
        return "N/A"
    hours, minutes, seconds = matches.groups()
    hours = int(hours) if hours else 0
    minutes = int(minutes) if minutes else 0
    seconds = int(seconds) if seconds else 0
    return f"{hours}:{minutes:02}:{seconds:02}" if hours else f"{minutes}:{seconds:02}"


def get_video_details(video_id):
    """
    指定された動画IDの詳細情報を取得する関数。
    """
    response = youtube.videos().list(
        part='snippet,contentDetails,statistics',
        id=video_id
    ).execute()

    if not response['items']:
        return None

    video = response['items'][0]
    snippet = video['snippet']
    content_details = video['contentDetails']
    statistics = video['statistics']

    video_data = {
        'video_id': video_id, # 動画ID
        'title': snippet.get('title', 'N/A'), # タイトル
        'description': snippet.get('description', 'N/A'), # 説明
        'thumbnail_url': snippet['thumbnails'].get('high', {}).get('url', 'N/A'), # サムネイルURL
        'duration': convert_duration(content_details.get('duration', 'N/A')), # 再生時間
        'view_count': statistics.get('viewCount', 'N/A'), # 再生回数
        'like_count': statistics.get('likeCount', 'N/A'), # いいね数
        'comment_count': statistics.get('commentCount', 'N/A'), # コメント数
        'channel_id': snippet.get('channelId', 'N/A'), # チャンネルID
        'channel_title': snippet.get('channelTitle', 'N/A'), # チャンネル名
    }
    return video_data


def get_channel_details(channel_id):
    """
    指定されたチャンネルIDの詳細情報を取得する関数。
    """
    response = youtube.channels().list(
        part='snippet,statistics',
        id=channel_id
    ).execute()

    if not response['items']:
        return None

    channel = response['items'][0]
    snippet = channel['snippet']
    statistics = channel['statistics']

    channel_data = {
        'channel_id': channel_id,
        'title': snippet.get('title', 'N/A'),
        'description': snippet.get('description', 'N/A'),
        'subscriber_count': statistics.get('subscriberCount', '非公開'),
        'channel_url': f"https://www.youtube.com/channel/{channel_id}"
    }
    return channel_data


def search_videos(keyword, max_results=10):
    """
    指定されたキーワードで動画を検索し、動画とチャンネルの詳細情報を取得する関数。
    """
    response = youtube.search().list(
        q=keyword,
        part='snippet',
        type='video',
        maxResults=max_results
    ).execute()

    video_data_list = []
    for item in response['items']:
        video_id = item['id']['videoId']
        video_data = get_video_details(video_id)
        if video_data:
            channel_id = video_data['channel_id']
            channel_data = get_channel_details(channel_id)
            if channel_data:
                video_data.update({
                    'channel_title': channel_data['title'], # チャンネル名
                    'channel_url': channel_data['channel_url'], # チャンネルURL
                    'subscriber_count': channel_data['subscriber_count'], # チャンネル登録者数
                    'channel_description': channel_data['description'] # チャンネル説明
                })
                video_data_list.append(video_data)
    return video_data_list


def save_to_csv(data, filename):
    """
    データをCSVファイルに保存する関数。
    """
    fieldnames = [
        "Video Title", "Video URL", "Thumbnail URL", "Video Description",
        "Video Duration", "Video Views", "Channel Title", "Channel URL",
        "Subscriber Count", "Channel Description"
    ]
    with open(filename, mode="w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for video in data:
            writer.writerow({
                "Video Title": video['title'],
                "Video URL": f"https://www.youtube.com/watch?v={video['video_id']}",
                "Thumbnail URL": video['thumbnail_url'],
                "Video Description": video['description'],
                "Video Duration": video['duration'],
                "Video Views": video['view_count'],
                "Channel Title": video['channel_title'],
                "Channel URL": video['channel_url'],
                "Subscriber Count": video['subscriber_count'],
                "Channel Description": video['channel_description']
            })


def get_popular_videos_from_channel(channel_id, max_results=5):
    """
    指定されたチャンネルから人気の動画を取得する関数。
    """
    # チャンネルのアップロード動画のプレイリストIDを取得
    response = youtube.channels().list(
        part='contentDetails',
        id=channel_id
    ).execute()

    if not response['items']:
        return []

    uploads_playlist_id = response['items'][0]['contentDetails']['relatedPlaylists']['uploads']

    # プレイリストから動画を取得
    videos = []
    next_page_token = None
    while len(videos) < max_results:
        playlist_response = youtube.playlistItems().list(
            part='snippet',
            playlistId=uploads_playlist_id,
            maxResults=50,
            pageToken=next_page_token
        ).execute()

        video_ids = [item['snippet']['resourceId']['videoId'] for item in playlist_response['items']]

        # 動画の詳細情報を取得
        videos_response = youtube.videos().list(
            part='snippet,contentDetails,statistics',
            id=','.join(video_ids)
        ).execute()

        for video in videos_response['items']:
            video_data = {
                'video_id': video['id'],
                'title': video['snippet']['title'],
                'description': video['snippet'].get('description', 'N/A'),
                'thumbnail_url': video['snippet']['thumbnails']['high']['url'],
                'duration': convert_duration(video['contentDetails']['duration']),
                'view_count': int(video['statistics'].get('viewCount', 0)),
                'like_count': video['statistics'].get('likeCount', 'N/A'),
                'comment_count': video['statistics'].get('commentCount', 'N/A')
            }
            videos.append(video_data)

        next_page_token = playlist_response.get('nextPageToken')
        if not next_page_token:
            break

    # 再生回数でソート
    videos.sort(key=lambda x: x['view_count'], reverse=True)

    return videos[:max_results]


def save2csv(data, filename, fieldnames):
    """
    データを指定されたフィールド名でCSVファイルに保存する関数。
    """
    with open(filename, mode="w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)


def search_main():
    # 検索キーワードを指定
    search_keyword = "ホロライブ　切り抜き"
    # 動画を検索
    videos = search_videos(search_keyword, max_results=100)
    formatted_videos = []
    for video in videos:
        formatted_videos.append({
            "Video Title": video.get("title", ""),
            "Video URL": f"https://www.youtube.com/watch?v={video.get('video_id', '')}",
            "Thumbnail URL": video.get("thumbnail_url", ""),
            "Video Description": video.get("description", ""),
            "Video Duration": video.get("duration", ""),
            "Video Views": video.get("view_count", ""),
            "Channel Title": video.get("channel_title", ""),
            "Channel URL": video.get("channel_url", ""),
            "Subscriber Count": video.get("subscriber_count", ""),
            "Channel Description": video.get("channel_description", "")
        })

    fieldnames = [
        "Video Title",
        "Video URL",
        "Thumbnail URL",
        "Video Description",
        "Video Duration",
        "Video Views",
        "Channel Title",
        "Channel URL",
        "Subscriber Count",
        "Channel Description"
    ]
    # CSVファイルに保存
    # 現在時刻を取得し、フォーマットする
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    csv_filename = f"{search_keyword}_{current_time}_videos.csv"
    save2csv(formatted_videos, csv_filename, fieldnames)  # 修正点：formatted_videosを渡す

    print(f"検索結果データが {csv_filename} に保存されました。")


# チャンネルの人気動画を取得
def get_channel_csv():
    """
    channel_list.csv を読み込んで Channel Description 部分を取得し、
    prompt_clip_api_call を使って処理を実行する関数。
    """
    file_name = "ホロライブ　切り抜き_2024-11-16_18-26-28_videos.csv"

    try:
        # CSVファイルを読み込む
        df = pd.read_csv(file_name, encoding="utf-8-sig")

        # 'Channel Description'カラムが存在するか確認
        if "Channel Description" not in df.columns:
            print("CSVファイルに 'Channel Description' カラムが見つかりませんでした。")
            return

        # 'Channel Description'カラムの最初の1件を取得
        first_description = df["Channel Description"].iloc[1]

        print(first_description)
        # prompt_clip_api_call 関数で使用
        result = prompt_clip_api_call(first_description)

        # 結果を表示
        print("API呼び出し結果:", result)

    except FileNotFoundError:
        print(f"{file_name} が見つかりませんでした。")
    except Exception as e:
        print("エラーが発生しました:", str(e))

def get_popular_main():
    # チャンネルの人気動画を取得
    channel_id = "UCs6nmQViDpUw0nuIx9c_WvA"  # ProgrammingKnowledgeのチャンネルID
    popular_videos = get_popular_videos_from_channel(channel_id, max_results=5)

    if popular_videos:
        # チャンネル名を取得
        channel_details = get_channel_details(channel_id)
        channel_name = channel_details['title'] if channel_details else "UnknownChannel"

        # ファイル名を「チャンネルID_チャンネル名.csv」とする
        sanitized_channel_name = re.sub(r'[\\/*?:"<>|]', "", channel_name)  # ファイル名に使用できない文字を除去
        csv_filename_popular = f"{channel_id}_{sanitized_channel_name}.csv"

        # 人気動画をCSVに保存
        popular_fieldnames = [
            "Video Title", "Video URL", "Video Description", "Video Duration",
            "Video Views", "Likes", "Comments", "Thumbnail URL"
        ]
        save2csv(popular_videos, csv_filename_popular, popular_fieldnames)
        print(f"人気動画データが {csv_filename_popular} に保存されました。")
    else:
        print("人気動画の取得に失敗しました。")

if __name__ == "__main__":
    # search_main()
    get_channel_csv()

    # get_popular_main()
