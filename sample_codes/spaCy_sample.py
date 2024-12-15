import spacy
import pandas as pd
from concurrent.futures import ThreadPoolExecutor

# GPUの使用設定（必要なら有効化、不要ならコメントアウト）
spacy.require_gpu()

# SpaCyモデルのパス
MODEL_PATH = "ja_core_news_sm"


def split_text_to_fit_limit(text, byte_limit=49149):
    """
    SudachiPyの制限を考慮し、バイト数が一定値を超えないようにテキストを分割。
    """
    chunks = []
    current_chunk = ""
    for char in text:
        # 次の文字を追加してバイト数超過するかチェック
        if len(current_chunk.encode('utf-8')) + len(char.encode('utf-8')) > byte_limit:
            if current_chunk:
                chunks.append(current_chunk)
            current_chunk = char
        else:
            current_chunk += char
    if current_chunk:
        chunks.append(current_chunk)
    return chunks


def process_batch(batch, leftover_text=""):
    """
    与えられたバッチを処理し、文単位でセグメントを出力する。
    leftover_text が渡された場合、それをバッチ先頭に追加することでバッチをまたぐテキスト結合を実現。

    Returns:
        (list, str):
           list: 完成した文のセグメント辞書のリスト
           str:  未完了（最後に文として確定できなかった）テキスト
    """
    nlp = spacy.load(MODEL_PATH)

    # バッチ内で連続するテキストとタイムスタンプを統合
    # leftover_textがある場合は先頭に付与
    full_text = leftover_text
    start_times = []
    end_times = []
    audio_path = None

    for row in batch:
        text = row['text']
        start = row['start']
        end = row['end']
        apath = row['audio_path']

        if audio_path is None:
            audio_path = apath

        # セグメントを連結
        full_text += text
        start_times.append(start)
        end_times.append(end)

    # テキストを分割して処理（SudachiPyバイト制限対応）
    chunks = split_text_to_fit_limit(full_text)

    completed_segments = []
    leftover = ""

    # 処理した文章ごとにstart/endを割り当てるためのロジック
    # ここでは簡易的に、各文が全文中のどのあたりに対応するかは考慮せず、
    # バッチの最初と最後のタイムスタンプを文境界として扱う簡易例とする。
    # 必要であれば文ごとに対応する時間を再計算する処理を追加すること。
    # 以下はバッチ全体を一つのチャンクとして扱い、startは最初、endは最後を割り当てる。
    # （ユーザー要望により簡略化。文ごとに細かい時間を割り当てたい場合、追加ロジックが必要）
    start_time = start_times[0] if start_times else None
    end_time = end_times[-1] if end_times else None

    for c_i, chunk in enumerate(chunks):
        doc = nlp(chunk)
        sents = list(doc.sents)

        # もし leftover があり、そこから繋がる場合は先頭の文を繋げる
        if leftover:
            # leftover + 最初のsent.textが繋がるかどうかを判定する必要があるが
            # sentence segmentationは SpaCy が再度行うので leftover があれば
            # 既に前チャンクで確定できなかった文はここで再解析すべき。
            # シンプルに leftover と chunk を合わせて解析する手もあるが、
            # 今回は leftover は単純につなげてから再度 nlp してもよい。
            # ただし既にdocがあるため、ここでは簡略化として leftoverを直前chunkに
            # 合流させておくべきだった。（この段階まできているため以下簡易案）
            #
            # 簡易的対処: leftoverはchunkの先頭にconcatして再解析
            combined_chunk = leftover + chunk
            doc = nlp(combined_chunk)
            sents = list(doc.sents)
            leftover = ""  # leftoverは再解析で消費

        # 最後の文は未確定（完全な句読点で終わらない文）かもしれないので、
        # 全て確定として扱い、最後の文のみ leftover に残す。
        # ここでは SpaCy の sentence segmentation は基本的に文末で確定するため、
        # 「未完了文」が発生するのはテキスト末尾に句読点などが無い場合のみ。
        # その場合は最後の文は leftover とし、それ以外は確定。
        if sents:
            # 最後の文を一旦取り出す
            last_sent = sents.pop()
            # 残りは全て確定
            for sent in sents:
                completed_segments.append({
                    'text': sent.text.strip(),
                    'start': start_time,
                    'end': end_time,
                    'audio_path': audio_path
                })
            # 最後の文はテキスト末尾に終止符があるなら確定、なければ leftover
            # 簡易判定として、最後の文字が句読点(。！？)で終わっていれば確定、それ以外はleftover
            if last_sent.text.strip()[-1:] in ['。', '!', '？', '?']:
                # 終止符で終わる → 確定
                completed_segments.append({
                    'text': last_sent.text.strip(),
                    'start': start_time,
                    'end': end_time,
                    'audio_path': audio_path
                })
            else:
                # 終止符で終わらない → leftover
                leftover = last_sent.text.strip()

    return completed_segments, leftover


def combine_segments_with_timestamps_parallel(segments_df, batch_size=10):
    """
    並列処理でCSVファイルのテキストを文単位で結合。
    前バッチからの未完了分は leftover として次バッチに渡し、重複が発生しないようにする。
    """
    # DataFrameをバッチごとに分割
    batches = [
        segments_df.iloc[i:i + batch_size].to_dict(orient="records")
        for i in range(0, len(segments_df), batch_size)
    ]

    combined_results = []
    leftover_text = ""  # 前バッチからの未完了テキスト

    with ThreadPoolExecutor() as executor:
        for batch in batches:
            future = executor.submit(process_batch, batch, leftover_text)
            batch_result, leftover_text_next = future.result()

            # 確定した文をcombined_resultsに追加
            combined_results.extend(batch_result)

            # 次バッチ用に leftover_text を更新
            leftover_text = leftover_text_next

    # 最後に leftover_text が残っていたら、それは完結しない文なので無視するか、
    # あるいは最終的に何らかの形で出力したいならここで出力可能。
    # ここでは無視する（未完了だから出さない）。
    # もし出したいなら、以下のようにすることも可能：
    # if leftover_text:
    #     combined_results.append({
    #         'text': leftover_text.strip(),
    #         'start': None,
    #         'end': None,
    #         'audio_path': None
    #     })

    return pd.DataFrame(combined_results)


if __name__ == "__main__":
    input_csv_path = "../data/transcription/source/0a5lFZB7jf8.csv"
    output_csv_path = "./combined_segments.csv"

    df = pd.read_csv(input_csv_path)

    # セグメントを結合
    combined_segments_df = combine_segments_with_timestamps_parallel(df, batch_size=25)

    # 結果を保存
    combined_segments_df.to_csv(output_csv_path, index=False, encoding="utf-8-sig")
    print(f"結果を保存しました: {output_csv_path}")
