import spacy
import pandas as pd
from concurrent.futures import ThreadPoolExecutor

# GPUの使用設定
spacy.require_gpu()

# SpaCyモデルのパス
MODEL_PATH = "ja_core_news_sm"


def split_text_to_fit_limit(text, byte_limit=49149):
    """
    SudachiPy等の制限に対応するため、UTF-8バイト長で分割。
    """
    chunks = []
    current_chunk = ""
    for char in text:
        if len(current_chunk.encode('utf-8')) + len(char.encode('utf-8')) > byte_limit:
            chunks.append(current_chunk)
            current_chunk = ""
        current_chunk += char
    if current_chunk:
        chunks.append(current_chunk)
    return chunks


def process_batch(batch):
    """
    バッチ内のセグメントを文分割し、各文に対応するstart/endを再計算する。
    """
    if not batch:
        return []
    nlp = spacy.load(MODEL_PATH)

    # 全文テキストとセグメント境界情報を作成
    full_text = ""
    segment_boundaries = []  # (start_char, end_char, start_time, end_time, audio_path)
    current_offset = 0
    for seg in batch:
        seg_text = seg['text']
        start_char = current_offset
        end_char = current_offset + len(seg_text)
        segment_boundaries.append((start_char, end_char, seg['start'], seg['end'], seg['audio_path']))
        full_text += seg_text
        current_offset = end_char

    # 文分割処理（テキストが長い場合、分割して処理）
    sentence_offsets = []
    processed_offset = 0
    text_chunks = split_text_to_fit_limit(full_text)
    for chunk in text_chunks:
        doc = nlp(chunk)
        for sent in doc.sents:
            # chunk内オフセット -> full_textオフセットへ変換
            sent_start = processed_offset + sent.start_char
            sent_end = processed_offset + sent.end_char
            sentence_offsets.append((sent.text.strip(), sent_start, sent_end))
        processed_offset += len(chunk)

    results = []
    # 各文がどのセグメント範囲と重なっているか確認し、start/end計算
    for sent_text, s_start, s_end in sentence_offsets:
        overlapping_segments = []
        for (seg_start_char, seg_end_char, seg_start_time, seg_end_time, seg_audio_path) in segment_boundaries:
            # 文範囲 [s_start, s_end) とセグメント範囲 [seg_start_char, seg_end_char) の重なりを確認
            if seg_end_char > s_start and seg_start_char < s_end:
                overlapping_segments.append((seg_start_time, seg_end_time, seg_audio_path))

        if not overlapping_segments:
            # 重なるセグメントがない場合はスキップ（理論上起きないはず）
            continue

        # 文に対応するstart, end, audio_pathの決定
        start_time = min(s[0] for s in overlapping_segments)
        end_time = max(s[1] for s in overlapping_segments)
        audio_path = overlapping_segments[0][2]  # 最初のセグメントのaudio_pathを使用

        results.append({
            'text': sent_text,
            'start': start_time,
            'end': end_time,
            'audio_path': audio_path
        })

    return results


def combine_segments_with_timestamps_parallel(segments_df, batch_size=10):
    """
    並列処理でバッチ単位で文分割とタイムスタンプ調整を行う。
    """
    # DataFrameをバッチに分割
    batches = [
        segments_df.iloc[i:i + batch_size].to_dict(orient="records")
        for i in range(0, len(segments_df), batch_size)
    ]

    combined_results = []

    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_batch, batch) for batch in batches]
        for f in futures:
            combined_results.extend(f.result())

    return pd.DataFrame(combined_results)


if __name__ == "__main__":
    input_csv_path = "../data/transcription/source/0a5lFZB7jf8.csv"
    output_csv_path = "./combined_segments.csv"

    df = pd.read_csv(input_csv_path)

    combined_segments_df = combine_segments_with_timestamps_parallel(df, batch_size=25)
    combined_segments_df.to_csv(output_csv_path, index=False, encoding="utf-8-sig")
    print(f"結果を保存しました: {output_csv_path}")
