import openai
import os
from dotenv import load_dotenv

load_dotenv()

OpenAIKey = os.environ['OpenAIKey']

# OpenAI APIキーの設定
openai.api_key = OpenAIKey

def chat_with_gpt(prompt):
    response = openai.chat.completions.create(
        model="gpt-4o-mini",  # 使用するモデルを指定
        messages=[
            {"role": "system", "content": "あなたは有能なアシスタントです。"},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content.strip()

if __name__ == "__main__":
    user_input = input("ユーザー: ")
    response = chat_with_gpt(user_input)
    print(f"ChatGPT: {response}")
