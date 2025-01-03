import os
import base64
import requests
from openai import OpenAI
from dotenv import load_dotenv

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

        try:
            # 画像をダウンロード
            response = requests.get(image_url)
            response.raise_for_status()
            image_data = response.content

            # 画像をBase64エンコード
            encoded_image = base64.b64encode(image_data).decode('utf-8')

            # OpenAI APIへのリクエスト
            message = [
                {"role": "system", "content": "あなたはOCRマシンです。抽出した文字以外は回答しないでください。"},
                {"role": "user", "content": "以下の画像に書かれている文字を抽出してください。"},
                {"role": "user", "content": f"data:image/png;base64,{encoded_image}"}
            ]

            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=message,
                temperature=0.2,
            )
            extracted = response.choices[0].message.content.strip()
        except requests.exceptions.RequestException as e:
            print("画像のダウンロード中にエラーが発生しました:", str(e))
            extracted = "画像のダウンロード中にエラーが発生しました。"
        except Exception as e:
            print("エラーが発生しました:", str(e))
            extracted = "エラーが発生しました。"

        return extracted

if __name__ == "__main__":
    image_url = "https://yt3.ggpht.com/Co2M3WdCbJm7t2q5XZOgBb8a5DQ1PUQpFsF8QkVfWOd-5SRVpb5YyazcczbzQh8yUKOqzocS"
    result = ImageText().image2text(image_url)
    print("結果:", result)
