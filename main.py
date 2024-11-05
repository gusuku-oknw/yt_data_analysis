# これはサンプルの Python スクリプトです。
from chat_downloader import ChatDownloader
import pandas as pd
import tqdm
from FunctionUtils import FunctionUtils


# Shift+F10 を押して実行するか、ご自身のコードに置き換えてください。
# Shift を2回押す を押すと、クラス/ファイル/ツールウィンドウ/アクション/設定を検索します。


def chat_download_csv(url):
    # 抽出データのリスト
    messages_data = []

    try:
        chat = ChatDownloader().get_chat(url,
                                         # start_time='0:00',
                                         # end_time='0:01:00',
                                         message_groups=['messages', 'superchat']
                                         )
    # create a generator
    except Exception as e:
        print(f"Error during transcription: {e}")
        return None

    # --------------------------------------------------------------------
    # chat = "<chat_downloader.sites.common.Chat object at 0x0000027CC147BDF0>"
    for message in tqdm.tqdm(chat):  # iterate over messages

        print(f'message: {message}')
        # 各メッセージからデータを抽出
        message_text = message.get('message')
        amount = message.get('money', {}).get('amount')
        time_in_seconds = message.get('time_in_seconds')

        # 必要なデータを辞書形式でリストに追加
        messages_data.append({
            "Time_in_seconds": time_in_seconds,
            "Message": message_text,
            "Amount": amount,
        })

    # DataFrameに変換してCSVファイルとして出力
    df = pd.DataFrame(messages_data)
    df.to_csv('./chat_messages.csv', index=False, encoding='utf-8-sig')
    print("CSVファイル 'chat_messages.csv' が作成されました。")


# ガター内の緑色のボタンを押すとスクリプトを実行します。
if __name__ == '__main__':
    url = "https://www.youtube.com/live/c2mRKDgC7zY"
    chat_download_csv(url)

# PyCharm のヘルプは https://www.jetbrains.com/help/pycharm/ を参照してください
