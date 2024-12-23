import pandas as pd
import os
import yaml
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score
import re

# comparison_matchesを元にVideo IDを抽出する関数
def extract_video_ids_from_matches(data_dirs):
    """
    comparison_matchesディレクトリからVideo IDを抽出する。

    Parameters:
        data_dirs (dict): ディレクトリパスの辞書。

    Returns:
        set: 抽出されたVideo IDの集合。
    """
    matches_dir = data_dirs["comparison_matches"]
    video_ids = set()

    for file_name in os.listdir(matches_dir):
        if file_name.endswith("_matches.csv"):
            # ファイル名の正規表現パターン
            match = re.match(r"(.+?)_(.+?)_matches\.csv", file_name)
            if match:
                video_ids.add((match.group(1), match.group(2)))  # (Source ID, Clipping ID)

    return video_ids

# Source を基準に Matches を利用してデータを生成する関数
def process_source_with_matches(source_id, clipping_id, data_dirs):
    """
    Source を基準に Matches を利用してマッチ情報を付与する。

    Parameters:
        source_id (str): 元動画ID。
        clipping_id (str): 切り抜き動画ID。
        data_dirs (dict): ディレクトリパスの辞書。

    Returns:
        pd.DataFrame: マッチ情報を含むデータフレーム。
    """
    source_path = os.path.join(data_dirs["transcription_source"], f"{source_id}.csv")
    matches_path = os.path.join(data_dirs["comparison_matches"], f"{source_id}_{clipping_id}_matches.csv")

    # ファイルの存在と空チェック
    if not os.path.exists(source_path) or os.path.getsize(source_path) == 0:
        print(f"Source データが見つからないか空です: {source_path}")
        return pd.DataFrame()
    if not os.path.exists(matches_path) or os.path.getsize(matches_path) == 0:
        print(f"Matches データが見つからないか空です: {matches_path}")
        return pd.DataFrame()

    # ファイルの読み込み
    try:
        source_df = pd.read_csv(source_path)
    except pd.errors.EmptyDataError:
        print(f"Source データが空です: {source_path}")
        return pd.DataFrame()

    try:
        matches_df = pd.read_csv(matches_path)
    except pd.errors.EmptyDataError:
        print(f"Matches データが空です: {matches_path}")
        return pd.DataFrame()

    # 必要なカラムの確認
    required_columns_source = {"text", "start", "end"}
    required_columns_matches = {"source_text", "source_start", "source_end", "clip_text", "clip_start", "clip_end", "similarity"}

    if not required_columns_source.issubset(source_df.columns):
        print("Source データのカラムが不足しています。")
        return pd.DataFrame()
    if not required_columns_matches.issubset(matches_df.columns):
        print("Matches データのカラムが不足しています。")
        return pd.DataFrame()

    # Source に Matches を結合
    source_df = source_df.rename(columns={"text": "source_text", "start": "source_start", "end": "source_end"})
    source_df = source_df.drop(columns=[col for col in source_df.columns if col == "audio_path"], errors="ignore")

    # Source 全体を保持し、Matches を結合
    merged_df = pd.merge(source_df, matches_df, on=["source_text", "source_start", "source_end"], how="left")

    # is_match の設定
    merged_df["is_match"] = ~merged_df["clip_text"].isnull()

    # 動画ID情報を追加
    merged_df["source_video_id"] = source_id
    merged_df["clipping_video_id"] = clipping_id

    return merged_df

# すべての動画を処理してデータを統合する関数
def process_all_videos(config_path):
    """
    すべてのVideo_idについてデータを処理し、Train/Testデータを生成する。

    Parameters:
        config_path (str): 設定ファイル（YAML）のパス。
    """
    # 設定ファイルの読み込み
    with open(config_path, 'r', encoding='utf-8-sig') as f:
        config = yaml.safe_load(f)

    data_dirs = config["data_dirs"]

    # comparison_matchesを元にVideo IDを取得
    all_video_ids = extract_video_ids_from_matches(data_dirs)

    print(f"取得したVideoペアの数: {len(all_video_ids)}")

    all_data = []
    total_source_rows = 0

    for source_id, clipping_id in all_video_ids:
        print(f"Processing Source: {source_id}, Clipping: {clipping_id}")
        processed_df = process_source_with_matches(source_id, clipping_id, data_dirs)
        if not processed_df.empty:
            all_data.append(processed_df)
            total_source_rows += len(processed_df)

    print(f"総Sourceの行数: {total_source_rows}")

    if not all_data:
        print("No data processed.")
        return

    # データを統合し、source_video_id 内でソート
    combined_data = pd.concat(all_data, ignore_index=True)
    combined_data = combined_data.sort_values(by=["source_video_id", "source_start", "source_end", "clip_start", "clip_end"], ignore_index=True)

    # Train/Test分割をVideo ID単位で実施
    video_ids = combined_data["source_video_id"].unique()
    train_ids, test_ids = train_test_split(video_ids, test_size=0.2, random_state=42)

    train_data = combined_data[combined_data["source_video_id"].isin(train_ids)].reset_index(drop=True)
    test_data = combined_data[combined_data["source_video_id"].isin(test_ids)].reset_index(drop=True)

    # スコアの計算
    y_train = train_data["is_match"]
    y_test = test_data["is_match"]

    train_precision = precision_score(y_train, y_train)
    train_recall = recall_score(y_train, y_train)
    train_f1 = f1_score(y_train, y_train)

    test_precision = precision_score(y_test, y_test)
    test_recall = recall_score(y_test, y_test)
    test_f1 = f1_score(y_test, y_test)

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
