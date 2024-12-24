from datetime import datetime
from chat_downloader import ChatDownloader
import pandas as pd
from urllib.parse import urlparse, parse_qs

class ChatProcessor:
    @staticmethod
    def get_video_id_from_url(url):
        """
        Extract YouTube video ID from URL.
        """
        parsed_url = urlparse(url)

        if "youtube.com" in parsed_url.netloc:
            query_params = parse_qs(parsed_url.query)
            if "v" in query_params:
                return query_params["v"][0]
            if "/live/" in parsed_url.path:
                return parsed_url.path.split("/")[-1]

        if "youtu.be" in parsed_url.netloc:
            return parsed_url.path.strip("/")

        return f"unknown_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    @staticmethod
    def remove_query_params(url):
        """
        Remove query parameters from a URL, except for YouTube video ID.
        """
        parsed_url = urlparse(url)
        query_params = parse_qs(parsed_url.query)
        if "v" in query_params:
            base_url = f"{parsed_url.scheme}://{parsed_url.netloc}{parsed_url.path}"
            return f"{base_url}?v={query_params['v'][0]}"
        return f"{parsed_url.scheme}://{parsed_url.netloc}{parsed_url.path}"

    def download_chat(self, url):
        """
        Download chat messages from a given URL.
        Returns:
            List[dict]: A list of messages with time, message, and amount details.
        """
        messages_data = []
        try:
            chat = ChatDownloader().get_chat(
                url,
                message_groups=["messages", "superchat"]
            )
            for message in chat:
                messages_data.append({
                    "Time_in_seconds": message.get("time_in_seconds"),
                    "Message": message.get("message"),
                    "Amount": message.get("money", {}).get("amount"),
                })
        except Exception as e:
            print(f"Error during chat download: {e}")
        return messages_data

    def process_chat(self, url, handle_data):
        """
        Process chat data from a given URL and handle it using a custom function.

        Parameters:
            url (str): The URL to process.
            handle_data (function): A function that handles the processed data.
        """
        video_id = self.get_video_id_from_url(self.remove_query_params(url))
        chat_data = self.download_chat(url)
        if chat_data:
            handle_data(video_id, chat_data)
        else:
            print(f"No data found for video ID: {video_id}")


# Example of a custom handler to save data in memory or elsewhere
def save_data(video_id, data):
    """
    Custom handler example to save chat data.
    """
    print(f"Processing data for video ID: {video_id}")
    # Save data to a file, database, or in-memory structure
    df = pd.DataFrame(data)
    print(df.head())  # Example of further processing or analysis


if __name__ == "__main__":
    processor = ChatProcessor()
    url = "https://www.youtube.com/live/YGGLxywB3Tw?si=O5Aa-5KqFPqQD8Xd"
    processor.process_chat(url, save_data)
