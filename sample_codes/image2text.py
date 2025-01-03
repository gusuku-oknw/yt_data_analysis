import os
import base64
import requests
from openai import OpenAI
from dotenv import load_dotenv

class ImageText:
    def __init__(self):
        load_dotenv()
        self.client = OpenAI(api_key=os.environ['OpenAIKey'])

    @staticmethod
    def encode_image_from_url(image_url):
        try:
            # 画像をダウンロード
            response = requests.get(image_url)
            response.raise_for_status()
            image_data = response.content
            # Base64エンコード
            return base64.b64encode(image_data).decode("utf-8")
        except requests.exceptions.RequestException as e:
            print("画像のダウンロード中にエラーが発生しました:", str(e))
            return None

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

        # 画像をBase64エンコード
        encoded_image = self.encode_image_from_url(image_url)
        if not encoded_image:
            return "画像のダウンロードまたはエンコードに失敗しました。"

        # OpenAI APIへのプロンプト構築
        message = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "何がかかれていますか？"
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64, {encoded_image}",
                        },
                    },
                ],
            }
        ]

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=message,
                temperature=0.2,
            )
            extracted = response.choices[0].message.content.strip()
        except Exception as e:
            print("エラーが発生しました:", str(e))
            extracted = "エラーが発生しました。"

        return extracted

if __name__ == "__main__":
    # テスト用の画像URL
    image_url = "https://yt3.ggpht.com/Co2M3WdCbJm7t2q5XZOgBb8a5DQ1PUQpFsF8QkVfWOd-5SRVpb5YyazcczbzQh8yUKOqzocS"

    # インスタンスを作成
    image_text = ImageText()

    # encode_image_from_url メソッドのテスト
    encoded_image = image_text.encode_image_from_url(image_url)

    if encoded_image:
        print("Base64エンコード成功:")
        print(encoded_image[:100] + "...")  # 結果の最初の100文字を表示
    else:
        print("画像のダウンロードまたはエンコードに失敗しました。")

    image_url = "https://yt3.ggpht.com/Co2M3WdCbJm7t2q5XZOgBb8a5DQ1PUQpFsF8QkVfWOd-5SRVpb5YyazcczbzQh8yUKOqzocS"
    result = ImageText().image2text(image_url)
    print("結果:", result)
