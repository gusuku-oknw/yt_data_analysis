# これはサンプルの Python スクリプトです。
from datetime import datetime
from chat_downloader import ChatDownloader
import pandas as pd
import tqdm
import os
from urllib.parse import urlparse, parse_qs
from FunctionUtils import FunctionUtils


# Shift+F10 を押して実行するか、ご自身のコードに置き換えてください。
# Shift を2回押す を押すと、クラス/ファイル/ツールウィンドウ/アクション/設定を検索します。
def get_video_id_from_url(url):
    """
    URLからYouTube動画IDを抽出する。
    - 通常のYouTube URL (例: https://www.youtube.com/watch?v=...)
    - 短縮URL (例: https://youtu.be/...)
    - ライブ配信URL (例: https://www.youtube.com/live/...)

    Parameters:
        url (str): 処理対象のURL。

    Returns:
        str: 抽出した動画ID。抽出できない場合は 'unknown_video_id' を返す。
    """
    parsed_url = urlparse(url)

    # YouTubeの通常の動画URL
    if "youtube.com" in parsed_url.netloc:
        query_params = parse_qs(parsed_url.query)
        if "v" in query_params:
            return query_params["v"][0]  # v= のパラメータから動画IDを取得
        # Live配信URL形式の場合
        if "/live/" in parsed_url.path:
            return os.path.basename(parsed_url.path)  # パスの最後の要素を動画IDと見なす

    # YouTubeの短縮URL (youtu.be形式)
    if "youtu.be" in parsed_url.netloc:
        return parsed_url.path.strip("/")  # パス部分が動画ID

    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    return f"{current_time}_unknown_video_id"  # 未対応の形式の場合


def remove_query_params(url):
    """
    クエリパラメータを除去したURLを返す。

    Parameters:
        url (str): 元のURL。

    Returns:
        str: クエリパラメータを除去したURL。
    """
    parsed_url = urlparse(url)
    clean_url = f"{parsed_url.scheme}://{parsed_url.netloc}{parsed_url.path}"
    return clean_url

def chat_download_csv(url, directory=None):
    # 抽出データのリスト
    messages_data = []

    try:
        chat = ChatDownloader().get_chat(url,
                                         # start_time='0:00',
                                         # end_time='0:10:00',
                                         message_groups=['messages', 'superchat']
                                         )
    # create a generator
    except Exception as e:
        print(f"Error during transcription: {e}")
        return None

    # --------------------------------------------------------------------
    # chat = "<chat_downloader.sites.common.Chat object at 0x0000027CC147BDF0>"
    for message in tqdm.tqdm(chat):  # iterate over messages

        # print(f'message: {message}')
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
    file_name = get_video_id_from_url(remove_query_params(url))

    if directory is not None:
        try:
            df.to_csv(f'./{directory}/{file_name}.csv', index=False, encoding='utf-8-sig')
        except FileNotFoundError:
            print(f"ディレクトリ '{directory}' が存在しません。")
    else:
        df.to_csv(f'./{file_name}.csv', index=False, encoding='utf-8-sig')
    print(f"CSVファイル '{file_name}.csv' が作成されました。")


# ガター内の緑色のボタンを押すとスクリプトを実行します。
if __name__ == '__main__':
    url = "https://www.youtube.com/watch?v=HaHgd60bWvQ&t=0s"
    chat_download_csv(url)

# PyCharm のヘルプは https://www.jetbrains.com/help/pycharm/ を参照してください
