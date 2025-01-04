import re
from urllib.parse import urlparse, parse_qs
from datetime import datetime

# =============================================================================
# 1) URLや動画IDを扱うユーティリティクラス
# =============================================================================
class YTURLUtils:
    @staticmethod
    def split_urls(row, allow_channels=False):
        url_pattern = re.compile(r'https?://[^\s,]+')
        urls = url_pattern.findall(row)
        return YTURLUtils.filter_and_correct_urls(urls, allow_channels=allow_channels)

    @staticmethod
    def filter_and_correct_urls(url_list, allow_playlists=False, allow_channels=False, exclude_twitch=True):
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
        parsed_url = urlparse(url)
        if "youtube.com" in parsed_url.netloc:
            query_params = parse_qs(parsed_url.query)
            if "v" in query_params:
                return query_params["v"][0]
            if "/live/" in parsed_url.path:
                return parsed_url.path.split("/")[-1]
        if "youtu.be" in parsed_url.netloc:
            return parsed_url.path.strip("/")
        current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        return f"{current_time}_unknown_video_id"

    @staticmethod
    def remove_query_params(url):
        parsed_url = urlparse(url)
        query_params = parse_qs(parsed_url.query)
        if "v" in query_params:
            base_url = f"{parsed_url.scheme}://{parsed_url.netloc}{parsed_url.path}"
            return f"{base_url}?v={query_params['v'][0]}"
        else:
            return f"{parsed_url.scheme}://{parsed_url.netloc}{parsed_url.path}"
