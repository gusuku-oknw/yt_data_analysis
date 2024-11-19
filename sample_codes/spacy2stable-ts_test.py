import spacy
from stable_whisper import modify_model, load_model
import json

# spaCyの日本語モデルをロード (GPU対応のトランスフォーマーモデル)
nlp = spacy.load("ja_core_news_trf")

def transcribe_with_timestamps(audio_path, whisper_model="large-v3"):
    """
    stable-tsを使って文字起こしとタイムスタンプを取得します。
    """
    model = load_model(whisper_model)
    modify_model(model)

    # Whisperで音声を文字起こし (stable-tsがタイムスタンプを含む結果を返します)
    result = model.transcribe(audio_path, language="ja", task="transcribe", word_timestamps=True)
    return result

def split_sentences_with_timestamps(segments, nlp_model):
    """
    文字起こし結果を文単位で分割し、タイムスタンプを保持します。

    Parameters:
        segments (list): stable-tsのセグメント情報。
        nlp_model: spaCyのモデル。

    Returns:
        list: 文単位のタイムスタンプ付き結果。
    """
    # 全体の文字起こし結果を結合
    full_text = "".join([seg.text for seg in segments])  # seg["text"]ではなくseg.text
    words = segments  # セグメント情報

    # spaCyで文単位に分割
    doc = nlp_model(full_text)
    sentence_segments = []

    for sent in doc.sents:
        sentence_text = sent.text
        start_time, end_time = None, None

        # 文中のタイムスタンプを計算
        for word_info in words:
            if word_info.text in sentence_text:  # seg["text"]ではなくseg.text
                if start_time is None or word_info.start < start_time:
                    start_time = word_info.start
                if end_time is None or word_info.end > end_time:
                    end_time = word_info.end

        sentence_segments.append({
            "sentence": sentence_text,
            "start": start_time,
            "end": end_time,
        })

    return sentence_segments

if __name__ == "__main__":
    audio_path = "../data/sound/clipping_audio_wav/htdemucs/7-1fNxXj_xM/vocals.wav"  # 入力音声ファイルのパス

    # ステップ1: 音声を文字起こしし、タイムスタンプを取得
    print("音声の文字起こし中...")
    transcription_result = transcribe_with_timestamps(audio_path)

    # # 結果をJSONファイルとして保存 (オプション)
    # with open("transcription_result.json", "w", encoding="utf-8") as f:
    #     json.dump(transcription_result, f, ensure_ascii=False, indent=2)

    # ステップ2: 文字起こし結果を文単位で分割
    print("文単位で分割中...")
    sentences_with_timestamps = split_sentences_with_timestamps(transcription_result, nlp)

    # 結果を表示
    for i, sent in enumerate(sentences_with_timestamps):
        start_time = f"{sent['start']:.2f}s" if sent['start'] is not None else "N/A"
        end_time = f"{sent['end']:.2f}s" if sent['end'] is not None else "N/A"

        print(f"文{i + 1}: {sent['sentence']}")
        print(f"  開始: {start_time}, 終了: {end_time}")
        print()
