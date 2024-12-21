import pandas as pd
import os
import yaml
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score

# 対応関係をチェックする関数
def check_file_consistency(video_ids, data_dirs):
    """
    各ディレクトリに対応するファイルが揃っているか確認する。

    Parameters:
        video_ids (list): チャットメッセージのVideo IDリスト。
        data_dirs (dict): ディレクトリパスの辞書。

    Returns:
        missing_files (dict): ディレクトリごとに不足しているファイルの情報。
    """
    missing_files = {}

    for dir_name, dir_path in data_dirs.items():
        if dir_name == "chat_messages":
            continue  # chat_messagesは基準として扱う

        missing_in_dir = []
        for video_id in video_ids:
            file_path = os.path.join(dir_path, f"{video_id}.csv")
            if not os.path.exists(file_path):
                missing_in_dir.append(video_id)

        if missing_in_dir:
            missing_files[dir_name] = missing_in_dir

    return missing_files

# チャットデータから切り抜きセグメントを推測する関数
def predict_clipping_segments(video_id, data_dirs, similarity_threshold=0.7):
    """
    チャットメッセージデータと元動画の文字起こしデータを比較し、
    切り抜かれる可能性が高いセグメントを推測する。

    Parameters:
        video_id (str): 動画ID。
        data_dirs (dict): ディレクトリパスの辞書。
        similarity_threshold (float): 類似度スコアの閾値。

    Returns:
        pd.DataFrame: 推測された切り抜きセグメント。
    """
    # チャットメッセージの読み込み
    chat_path = os.path.join(data_dirs["chat_messages"], f"{video_id}.csv")
    if not os.path.exists(chat_path):
        print(f"チャットデータが見つかりません: {chat_path}")
        return pd.DataFrame()
    chat_df = pd.read_csv(chat_path)

    # 元動画の文字起こしデータの読み込み
    source_path = os.path.join(data_dirs["transcription_source"], f"{video_id}.csv")
    if not os.path.exists(source_path):
        print(f"元動画文字起こしデータが見つかりません: {source_path}")
        return pd.DataFrame()
    source_df = pd.read_csv(source_path)

    # 必要なカラムの確認
    if "Message" not in chat_df.columns or "text" not in source_df.columns:
        print("必要なカラムが不足しています。")
        return pd.DataFrame()

    # TF-IDFベクトル化
    vectorizer = TfidfVectorizer()
    source_texts = source_df["text"].fillna("").tolist()
    chat_messages = chat_df["Message"].fillna("").tolist()

    source_tfidf = vectorizer.fit_transform(source_texts)
    chat_tfidf = vectorizer.transform(chat_messages)

    # 類似度計算
    similarities = cosine_similarity(chat_tfidf, source_tfidf)

    # 推測されたセグメントを保持するリスト
    predicted_segments = []

    for chat_idx, chat_row in chat_df.iterrows():
        for source_idx, similarity in enumerate(similarities[chat_idx]):
            if similarity >= similarity_threshold:
                predicted_segments.append({
                    "chat_message": chat_row["Message"],
                    "chat_time": chat_row["Time_in_seconds"],
                    "source_text": source_df.iloc[source_idx]["text"],
                    "source_start": source_df.iloc[source_idx]["start"],
                    "source_end": source_df.iloc[source_idx]["end"],
                    "similarity": similarity
                })

    # DataFrameに変換
    predicted_df = pd.DataFrame(predicted_segments)

    return predicted_df

# すべての動画を処理してデータを統合する関数
def process_all_videos(config_path):
    """
    すべてのVideo_idについて切り抜きセグメントを推測し、
    Train/Testデータを生成してスコアを計算する。

    Parameters:
        config_path (str): 設定ファイル（YAML）のパス。
    """
    # 設定ファイルの読み込み
    with open(config_path, 'r', encoding='utf-8-sig') as f:
        config = yaml.safe_load(f)

    data_dirs = config["data_dirs"]
    similarity_threshold = config["similarity_threshold"]

    all_video_ids = [f.replace(".csv", "") for f in os.listdir(data_dirs["chat_messages"]) if f.endswith(".csv")]

    # ファイル対応関係の確認
    missing_files = check_file_consistency(all_video_ids, data_dirs)
    if missing_files:
        print("以下のファイルが不足しています:")
        for dir_name, missing in missing_files.items():
            print(f"{dir_name}: {missing}")
        return

    all_data = []
    for video_id in all_video_ids:
        print(f"Processing video_id: {video_id}")
        predicted_segments = predict_clipping_segments(video_id, data_dirs, similarity_threshold)
        if not predicted_segments.empty:
            predicted_segments["video_id"] = video_id
            all_data.append(predicted_segments)

    if not all_data:
        print("No data processed.")
        return

    # データを統合
    combined_data = pd.concat(all_data, ignore_index=True)

    # Train/Test分割
    train_data, test_data = train_test_split(combined_data, test_size=0.2, random_state=42)

    # スコアの計算
    y_train = train_data["similarity"] >= similarity_threshold
    y_test = test_data["similarity"] >= similarity_threshold

    y_train_pred = train_data["similarity"] >= similarity_threshold
    y_test_pred = test_data["similarity"] >= similarity_threshold

    train_precision = precision_score(y_train, y_train_pred)
    train_recall = recall_score(y_train, y_train_pred)
    train_f1 = f1_score(y_train, y_train_pred)

    test_precision = precision_score(y_test, y_test_pred)
    test_recall = recall_score(y_test, y_test_pred)
    test_f1 = f1_score(y_test, y_test_pred)

    print("Train Scores:")
    print(f"Precision: {train_precision:.2f}, Recall: {train_recall:.2f}, F1: {train_f1:.2f}")

    print("Test Scores:")
    print(f"Precision: {test_precision:.2f}, Recall: {test_recall:.2f}, F1: {test_f1:.2f}")

    # 結果を保存
    train_data.to_csv(config["train_files"], index=False, encoding="utf-8-sig")
    test_data.to_csv(config["test_files"], index=False, encoding="utf-8-sig")

    print("Train/Test data saved.")

# 実行例
if __name__ == "__main__":
    process_all_videos("config/default_config.yaml")
