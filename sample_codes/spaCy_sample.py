import spacy


def resegment_sentences_with_timestamps(segments, nlp):
    """
    SpaCyを使ってテキストを文単位で再分割し、新しいstartとendを計算する。

    Parameters:
        segments (list): [{'text': str, 'start': float, 'end': float, 'audio_path': str}, ...]
        nlp: SpaCyモデル

    Returns:
        list: [{'text': str, 'start': float, 'end': float, 'audio_path': str}, ...]
    """
    # 全体のテキストを結合
    full_text = " ".join([seg['text'] for seg in segments])

    # 全体のタイムスタンプのリストを作成
    timestamps = []
    for seg in segments:
        timestamps.append((seg['start'], seg['end']))

    # SpaCyで文ごとに分割
    doc = nlp(full_text)
    sentences = [sent.text for sent in doc.sents]

    # 新しい文ごとにタイムスタンプを計算
    new_segments = []
    total_length = sum([len(seg['text']) for seg in segments])  # 全体の文字数
    char_to_time_ratio = sum([seg['end'] - seg['start'] for seg in segments]) / total_length  # 1文字あたりの平均時間

    current_start = segments[0]['start']  # 初期値として最初のセグメントの開始時刻

    for sentence in sentences:
        sentence_length = len(sentence)
        sentence_duration = sentence_length * char_to_time_ratio
        new_end = current_start + sentence_duration

        new_segments.append({
            'text': sentence.strip(),
            'start': current_start,
            'end': new_end,
            'audio_path': segments[0]['audio_path']  # 同じファイルパスを使用
        })

        current_start = new_end  # 次の文の開始時刻を更新

    return new_segments


# 使用例
if __name__ == "__main__":
    # 入力セグメント（息継ぎで区切られたデータ）
    segments = [
        {'text': 'ゲージを取得する', 'start': 2.714, 'end': 4.134, 'audio_path': 'path/to/segment_02.wav'},
        {'text': '相手に近い時避難時にゲージが少しずつたまるダメージ受けるとムラつくんだね確かにマリンドエムだからね',
         'start': 4.186, 'end': 11.526, 'audio_path': 'path/to/segment_03.wav'},
        {'text': 'ブラムラしちゃおっかな', 'start': 12.058, 'end': 14.118, 'audio_path': 'path/to/segment_04.wav'},
        {'text': 'ムラムラゲージどこにある?', 'start': 14.97, 'end': 17.03, 'audio_path': 'path/to/segment_05.wav'},
        {'text': '左下にあるわえやばいちょっとずつムラムラしてきた', 'start': 17.466, 'end': 21.19,
         'audio_path': 'path/to/segment_06.wav'},
    ]

    # SpaCyモデルのロード
    nlp = spacy.load("ja_core_news_trf")

    # 文単位で再分割
    new_segments = resegment_sentences_with_timestamps(segments, nlp)

    # 結果を表示
    for seg in new_segments:
        print(seg)
