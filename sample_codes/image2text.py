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

        # OpenAI APIへのプロンプト構築
        message = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "この画像にかかれている文字と感情を抽出してください。なかった場合なんと言っていそうですか？\n"
                                             "[Joy, Sadness, Anticipation, Surprise, Anger, Fear, Disgust, Trust]の中で選んでください。テキストのみで出力してください。\n"
                                             "例: Hello, World!: Joy\n"
                                             "None: Anger"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_url,
                        },
                    },
                ],
            }
        ]

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
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
    image_url = "https://yt3.ggpht.com/f8WJ7Hw-3pO2kueey5WeySLC4fYDlTc3iXDv1Et18qC1ZXQ6QExbKBQPyflj_TMCNlSli9r1"

    # インスタンスを作成
    image_text = ImageText()

    result = ImageText().image2text(image_url)
    print("結果:", result)
