from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from difflib import SequenceMatcher


def compare_segments(clipping_segments, source_segments, initial_threshold=1.0, time_margin=30.0):
    """
    切り抜き動画と元動画を一致させる（並列処理を使用）。

    Parameters:
        clipping_segments (list): 切り抜き動画のセグメントリスト。
        source_segments (list): 元動画のセグメントリスト。
        initial_threshold (float): 初期の類似度しきい値。
        time_margin (float): 探索範囲の時間（秒）。

    Returns:
        list: 一致したセグメントペアのリスト。
    """
    def calculate_similarity(text1, text2, method="sequence"):
        """テキスト間の類似度を計算"""
        if method == "sequence":
            return SequenceMatcher(None, text1, text2).ratio()
        elif method == "jaccard":
            set1 = set(text1.split())
            set2 = set(text2.split())
            intersection = len(set1 & set2)
            union = len(set1 | set2)
            return intersection / union if union != 0 else 0
        elif method == "cosine":
            vectorizer = CountVectorizer().fit_transform([text1, text2])
            vectors = vectorizer.toarray()
            return cosine_similarity(vectors)[0, 1]
        else:
            raise ValueError(f"Unknown method: {method}")

    def find_best_match(segment, source_segments, threshold=0.8):
        """切り抜きセグメントに最も類似する元動画セグメントを探す"""
        best_match = None
        max_similarity = 0

        for src in source_segments:
            similarity = calculate_similarity(segment["text"], src["text"])
            if similarity > max_similarity and similarity >= threshold:
                best_match = src
                max_similarity = similarity

        return best_match

    def process_clip(clip):
        """単一のクリップを処理"""
        threshold = initial_threshold
        while threshold > 0.5:  # 最低しきい値まで試行
            # 探索範囲を秒単位で絞り込み
            search_range_start = max(0, clip["start"] - time_margin)
            search_range_end = min(source_segments[-1]["end"], clip["end"] + time_margin)

            # 秒単位をインデックス範囲に変換
            filtered_segments = [
                src for src in source_segments
                if search_range_start <= src["start"] <= search_range_end
            ]

            best_match = find_best_match(clip, filtered_segments, threshold)

            if best_match:
                return {
                    "clip_text": clip["text"],
                    "clip_start": clip["start"],
                    "clip_end": clip["end"],
                    "source_text": best_match["text"],
                    "source_start": best_match["start"],
                    "source_end": best_match["end"],
                    "similarity": calculate_similarity(clip["text"], best_match["text"]),
                }

            # 一致が見つからなければしきい値を下げる
            threshold -= 0.1
        return None

    # 並列処理の実行
    matches = []
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_clip, clip) for clip in clipping_segments]
        for future in as_completed(futures):
            result = future.result()
            if result is not None:
                matches.append(result)

    return matches



clipping_segments = [{'text': 'ムラムラ', 'start': 0.0, 'end': 0.934, 'audio_path': 'C:\\Users\\tmkjn\\Documents\\python\\data_analysis\\data_analysis\\content\\audio_segments_1732030601\\segment_00.wav'},
{'text': 'ムラムラゲームラ?', 'start': 1.978, 'end': 2.854, 'audio_path': 'C:\\Users\\tmkjn\\Documents\\python\\data_analysis\\data_analysis\\content\\audio_segments_1732030601\\segment_01.wav'},
{'text': 'ゲージを取得する', 'start': 2.714, 'end': 4.134, 'audio_path': 'C:\\Users\\tmkjn\\Documents\\python\\data_analysis\\data_analysis\\content\\audio_segments_1732030601\\segment_02.wav'},
{'text': '相手に近い時避難時にゲージが少しずつたまるダメージ受けるとムラつくんだね確かにマリンドエムだからね', 'start': 4.186, 'end': 11.526, 'audio_path': 'C:\\Users\\tmkjn\\Documents\\python\\data_analysis\\data_analysis\\content\\audio_segments_1732030601\\segment_03.wav'},
{'text': 'ブラムラしちゃおっかな', 'start': 12.058, 'end': 14.118, 'audio_path': 'C:\\Users\\tmkjn\\Documents\\python\\data_analysis\\data_analysis\\content\\audio_segments_1732030601\\segment_04.wav'},
{'text': 'ムラムラゲージどこにある?', 'start': 14.97, 'end': 17.029999999999998, 'audio_path': 'C:\\Users\\tmkjn\\Documents\\python\\data_analysis\\data_analysis\\content\\audio_segments_1732030601\\segment_05.wav'},
{'text': '左下にあるわえやばいちょっとずつムラムラしてきた', 'start': 17.466, 'end': 21.19, 'audio_path': 'C:\\Users\\tmkjn\\Documents\\python\\data_analysis\\data_analysis\\content\\audio_segments_1732030601\\segment_06.wav'},
{'text': 'ちょっとずつ', 'start': 21.466, 'end': 22.982, 'audio_path': 'C:\\Users\\tmkjn\\Documents\\python\\data_analysis\\data_analysis\\content\\audio_segments_1732030601\\segment_07.wav'},
{'text': 'これあれじゃん敵の中に', 'start': 24.89, 'end': 27.206, 'audio_path': 'C:\\Users\\tmkjn\\Documents\\python\\data_analysis\\data_analysis\\content\\audio_segments_1732030601\\segment_08.wav'},
{'text': 'モザイクかかってたよな', 'start': 29.69, 'end': 31.782, 'audio_path': 'C:\\Users\\tmkjn\\Documents\\python\\data_analysis\\data_analysis\\content\\audio_segments_1732030601\\segment_09.wav'},
{'text': 'おもろすぎるでしょちょっと', 'start': 31.802000000000003, 'end': 34.534000000000006, 'audio_path': 'C:\\Users\\tmkjn\\Documents\\python\\data_analysis\\data_analysis\\content\\audio_segments_1732030601\\segment_10.wav'},
{'text': 'こういう敵の中にバッて入っていったらマリンが一気にムラムラするんだなぁ', 'start': 34.617999999999995, 'end': 38.63, 'audio_path': 'C:\\Users\\tmkjn\\Documents\\python\\data_analysis\\data_analysis\\content\\audio_segments_1732030601\\segment_11.wav'},
{'text': 'カジシの道具来たねこれちょっと撮りたい', 'start': 39.45, 'end': 42.342000000000006, 'audio_path': 'C:\\Users\\tmkjn\\Documents\\python\\data_analysis\\data_analysis\\content\\audio_segments_1732030601\\segment_12.wav'},
{'text': 'これ金床2回使えるようになるから結構お得なんだよね', 'start': 42.938, 'end': 47.334, 'audio_path': 'C:\\Users\\tmkjn\\Documents\\python\\data_analysis\\data_analysis\\content\\audio_segments_1732030601\\segment_13.wav'},
{'text': '音症', 'start': 49.85, 'end': 51.398, 'audio_path': 'C:\\Users\\tmkjn\\Documents\\python\\data_analysis\\data_analysis\\content\\audio_segments_1732030601\\segment_14.wav'},
{'text': 'ああ', 'start': 51.098, 'end': 51.846, 'audio_path': 'C:\\Users\\tmkjn\\Documents\\python\\data_analysis\\data_analysis\\content\\audio_segments_1732030601\\segment_15.wav'},
{'text': 'ブラブラしてきました、君たち', 'start': 51.866, 'end': 54.438, 'audio_path': 'C:\\Users\\tmkjn\\Documents\\python\\data_analysis\\data_analysis\\content\\audio_segments_1732030601\\segment_16.wav'},
{'text': 'このほとばしるマリンの', 'start': 55.13, 'end': 57.67, 'audio_path': 'C:\\Users\\tmkjn\\Documents\\python\\data_analysis\\data_analysis\\content\\audio_segments_1732030601\\segment_17.wav'},
{'text': 'この体を', 'start': 57.37, 'end': 59.142, 'audio_path': 'C:\\Users\\tmkjn\\Documents\\python\\data_analysis\\data_analysis\\content\\audio_segments_1732030601\\segment_18.wav'},
{'text': 'めちゃくちゃにしてください', 'start': 59.226, 'end': 61.094, 'audio_path': 'C:\\Users\\tmkjn\\Documents\\python\\data_analysis\\data_analysis\\content\\audio_segments_1732030601\\segment_19.wav'},
{'text': 'バーネー', 'start': 62.65, 'end': 63.622, 'audio_path': 'C:\\Users\\tmkjn\\Documents\\python\\data_analysis\\data_analysis\\content\\audio_segments_1732030601\\segment_20.wav'},
{'text': 'ぽー', 'start': 67.45, 'end': 68.71000000000001, 'audio_path': 'C:\\Users\\tmkjn\\Documents\\python\\data_analysis\\data_analysis\\content\\audio_segments_1732030601\\segment_21.wav'},
{'text': 'わみつい', 'start': 69.36999999999999, 'end': 70.694, 'audio_path': 'C:\\Users\\tmkjn\\Documents\\python\\data_analysis\\data_analysis\\content\\audio_segments_1732030601\\segment_22.wav'},
{'text': 'ムラムラしてシールを飛ばしまっているよ今のマリンが', 'start': 70.39399999999999, 'end': 75.238, 'audio_path': 'C:\\Users\\tmkjn\\Documents\\python\\data_analysis\\data_analysis\\content\\audio_segments_1732030601\\segment_23.wav'},
{'text': '僕のマリンが', 'start': 74.97, 'end': 76.87, 'audio_path': 'C:\\Users\\tmkjn\\Documents\\python\\data_analysis\\data_analysis\\content\\audio_segments_1732030601\\segment_24.wav'},
{'text': 'マリンとワミーツイのコラボレーションやばいのでは?', 'start': 77.36999999999999, 'end': 81.73400000000001, 'audio_path': 'C:\\Users\\tmkjn\\Documents\\python\\data_analysis\\data_analysis\\content\\audio_segments_1732030601\\segment_25.wav'},
{'text': '超汁飛ばしまくってるけど', 'start': 82.49, 'end': 84.742, 'audio_path': 'C:\\Users\\tmkjn\\Documents\\python\\data_analysis\\data_analysis\\content\\audio_segments_1732030601\\segment_26.wav'},
{'text': '汁っ気が', 'start': 84.634, 'end': 85.958, 'audio_path': 'C:\\Users\\tmkjn\\Documents\\python\\data_analysis\\data_analysis\\content\\audio_segments_1732030601\\segment_27.wav'},
{'text': '大変なことなってますよ', 'start': 86.52199999999999, 'end': 88.96600000000001, 'audio_path': 'C:\\Users\\tmkjn\\Documents\\python\\data_analysis\\data_analysis\\content\\audio_segments_1732030601\\segment_28.wav'},
{'text': 'ハハハ', 'start': 89.49799999999999, 'end': 90.63, 'audio_path': 'C:\\Users\\tmkjn\\Documents\\python\\data_analysis\\data_analysis\\content\\audio_segments_1732030601\\segment_29.wav'},
{'text': 'この組み合わせなかなかですね', 'start': 92.154, 'end': 95.11, 'audio_path': 'C:\\Users\\tmkjn\\Documents\\python\\data_analysis\\data_analysis\\content\\audio_segments_1732030601\\segment_30.wav'},
{'text': '苦労と向け', 'start': 95.002, 'end': 96.518, 'audio_path': 'C:\\Users\\tmkjn\\Documents\\python\\data_analysis\\data_analysis\\content\\audio_segments_1732030601\\segment_31.wav'},
{'text': 'ダメだ誘惑に打ちかて', 'start': 98.874, 'end': 100.87, 'audio_path': 'C:\\Users\\tmkjn\\Documents\\python\\data_analysis\\data_analysis\\content\\audio_segments_1732030601\\segment_32.wav'},
{'text': 'すいません', 'start': 102.138, 'end': 102.886, 'audio_path': 'C:\\Users\\tmkjn\\Documents\\python\\data_analysis\\data_analysis\\content\\audio_segments_1732030601\\segment_33.wav'},
{'text': 'かわいい', 'start': 104.346, 'end': 105.478, 'audio_path': 'C:\\Users\\tmkjn\\Documents\\python\\data_analysis\\data_analysis\\content\\audio_segments_1732030601\\segment_34.wav'},
{'text': 'はい', 'start': 105.274, 'end': 105.99, 'audio_path': 'C:\\Users\\tmkjn\\Documents\\python\\data_analysis\\data_analysis\\content\\audio_segments_1732030601\\segment_35.wav'},
{'text': '懐かしいレビュー配信のココちゃんがさ絵本', 'start': 105.85, 'end': 110.022, 'audio_path': 'C:\\Users\\tmkjn\\Documents\\python\\data_analysis\\data_analysis\\content\\audio_segments_1732030601\\segment_36.wav'},
{'text': '絵本調で動画作ってたの思い出すなあ', 'start': 110.938, 'end': 114.47, 'audio_path': 'C:\\Users\\tmkjn\\Documents\\python\\data_analysis\\data_analysis\\content\\audio_segments_1732030601\\segment_37.wav'},
{'text': 'ホロライブのドラゴン', 'start': 115.61, 'end': 117.286, 'audio_path': 'C:\\Users\\tmkjn\\Documents\\python\\data_analysis\\data_analysis\\content\\audio_segments_1732030601\\segment_38.wav'},
{'text': '大門じゃんこれ', 'start': 117.722, 'end': 119.526, 'audio_path': 'C:\\Users\\tmkjn\\Documents\\python\\data_analysis\\data_analysis\\content\\audio_segments_1732030601\\segment_39.wav'},
{'text': '倒した相手がランダムでいろいろ落とす', 'start': 119.258, 'end': 122.182, 'audio_path': 'C:\\Users\\tmkjn\\Documents\\python\\data_analysis\\data_analysis\\content\\audio_segments_1732030601\\segment_40.wav'},
{'text': '何いろいろって', 'start': 122.01, 'end': 123.686, 'audio_path': 'C:\\Users\\tmkjn\\Documents\\python\\data_analysis\\data_analysis\\content\\audio_segments_1732030601\\segment_41.wav'},
{'text': '触れるとそれを持ち上げ相手に投げつけ', 'start': 123.514, 'end': 126.854, 'audio_path': 'C:\\Users\\tmkjn\\Documents\\python\\data_analysis\\data_analysis\\content\\audio_segments_1732030601\\segment_42.wav'},
{'text': 'ルノ?', 'start': 126.65, 'end': 127.622, 'audio_path': 'C:\\Users\\tmkjn\\Documents\\python\\data_analysis\\data_analysis\\content\\audio_segments_1732030601\\segment_43.wav'},
{'text': 'ハハハハ', 'start': 127.386, 'end': 128.23, 'audio_path': 'C:\\Users\\tmkjn\\Documents\\python\\data_analysis\\data_analysis\\content\\audio_segments_1732030601\\segment_44.wav'},
{'text': 'これもヒートアクションの一つだろう', 'start': 128.31400000000002, 'end': 130.79, 'audio_path': 'C:\\Users\\tmkjn\\Documents\\python\\data_analysis\\data_analysis\\content\\audio_segments_1732030601\\segment_45.wav'},
{'text': 'これ', 'start': 130.68200000000002, 'end': 131.718, 'audio_path': 'C:\\Users\\tmkjn\\Documents\\python\\data_analysis\\data_analysis\\content\\audio_segments_1732030601\\segment_46.wav'},
{'text': 'ハハハ', 'start': 131.674, 'end': 132.486, 'audio_path': 'C:\\Users\\tmkjn\\Documents\\python\\data_analysis\\data_analysis\\content\\audio_segments_1732030601\\segment_47.wav'},
{'text': 'ねえ', 'start': 133.85000000000002, 'end': 134.66199999999998, 'audio_path': 'C:\\Users\\tmkjn\\Documents\\python\\data_analysis\\data_analysis\\content\\audio_segments_1732030601\\segment_48.wav'},
{'text': '竜がごとくで草なんだけど', 'start': 134.87400000000002, 'end': 138.24599999999998, 'audio_path': 'C:\\Users\\tmkjn\\Documents\\python\\data_analysis\\data_analysis\\content\\audio_segments_1732030601\\segment_49.wav'},
{'text': '50キロの握力', 'start': 139.418, 'end': 142.47, 'audio_path': 'C:\\Users\\tmkjn\\Documents\\python\\data_analysis\\data_analysis\\content\\audio_segments_1732030601\\segment_50.wav'},
{'text': 'ギュッキュ', 'start': 142.49, 'end': 143.91, 'audio_path': 'C:\\Users\\tmkjn\\Documents\\python\\data_analysis\\data_analysis\\content\\audio_segments_1732030601\\segment_51.wav'},
{'text': 'びっくりした今50キロの握力怖かったんだ', 'start': 148.09, 'end': 151.78199999999998, 'audio_path': 'C:\\Users\\tmkjn\\Documents\\python\\data_analysis\\data_analysis\\content\\audio_segments_1732030601\\segment_52.wav'},
{'text': '50キロの悪力怖い怖い怖い怖い怖い怖い怖い', 'start': 156.73000000000002, 'end': 160.486, 'audio_path': 'C:\\Users\\tmkjn\\Documents\\python\\data_analysis\\data_analysis\\content\\audio_segments_1732030601\\segment_53.wav'},
{'text': 'めっちゃ怖い', 'start': 160.858, 'end': 161.926, 'audio_path': 'C:\\Users\\tmkjn\\Documents\\python\\data_analysis\\data_analysis\\content\\audio_segments_1732030601\\segment_54.wav'},
{'text': 'やが', 'start': 162.042, 'end': 162.75799999999998, 'audio_path': 'C:\\Users\\tmkjn\\Documents\\python\\data_analysis\\data_analysis\\content\\audio_segments_1732030601\\segment_55.wav'},
{'text': 'マジのゴリランの手出てきてる', 'start': 168.09, 'end': 171.42999999999998, 'audio_path': 'C:\\Users\\tmkjn\\Documents\\python\\data_analysis\\data_analysis\\content\\audio_segments_1732030601\\segment_56.wav'},
{'text': '星読みたちの群れが', 'start': 172.92200000000005, 'end': 175.11, 'audio_path': 'C:\\Users\\tmkjn\\Documents\\python\\data_analysis\\data_analysis\\content\\audio_segments_1732030601\\segment_57.wav'},
{'text': 'あれかな?', 'start': 175.994, 'end': 177.06199999999998, 'audio_path': 'C:\\Users\\tmkjn\\Documents\\python\\data_analysis\\data_analysis\\content\\audio_segments_1732030601\\segment_58.wav'},
{'text': '昨日のライブの帰り道だったかもしれない', 'start': 177.306, 'end': 180.006, 'audio_path': 'C:\\Users\\tmkjn\\Documents\\python\\data_analysis\\data_analysis\\content\\audio_segments_1732030601\\segment_59.wav'},
{'text': 'ハハハハ', 'start': 179.834, 'end': 181.254, 'audio_path': 'C:\\Users\\tmkjn\\Documents\\python\\data_analysis\\data_analysis\\content\\audio_segments_1732030601\\segment_60.wav'},
{'text': '昨日のライブめっちゃよかったわ', 'start': 182.138, 'end': 184.902, 'audio_path': 'C:\\Users\\tmkjn\\Documents\\python\\data_analysis\\data_analysis\\content\\audio_segments_1732030601\\segment_61.wav'},
{'text': 'ルイと一緒に家で見てたんだけどさあるいんちで', 'start': 184.794, 'end': 187.814, 'audio_path': 'C:\\Users\\tmkjn\\Documents\\python\\data_analysis\\data_analysis\\content\\audio_segments_1732030601\\segment_62.wav'},
{'text': 'ブドウ館やばいよね', 'start': 188.66600000000005, 'end': 191.558, 'audio_path': 'C:\\Users\\tmkjn\\Documents\\python\\data_analysis\\data_analysis\\content\\audio_segments_1732030601\\segment_63.wav'},
{'text': 'やばい', 'start': 193.082, 'end': 195.59, 'audio_path': 'C:\\Users\\tmkjn\\Documents\\python\\data_analysis\\data_analysis\\content\\audio_segments_1732030601\\segment_64.wav'},
{'text': 'やばいね', 'start': 196.89, 'end': 198.95, 'audio_path': 'C:\\Users\\tmkjn\\Documents\\python\\data_analysis\\data_analysis\\content\\audio_segments_1732030601\\segment_65.wav'},
{'text': '僕が入った時はVTuberの文化は', 'start': 198.874, 'end': 202.534, 'audio_path': 'C:\\Users\\tmkjn\\Documents\\python\\data_analysis\\data_analysis\\content\\audio_segments_1732030601\\segment_66.wav'},
{'text': 'まだまだこれから発展していくぞって感じだったけど', 'start': 202.682, 'end': 207.398, 'audio_path': 'C:\\Users\\tmkjn\\Documents\\python\\data_analysis\\data_analysis\\content\\audio_segments_1732030601\\segment_67.wav'},
{'text': '今もそうなんだと思うんだけどさ', 'start': 207.418, 'end': 210.31, 'audio_path': 'C:\\Users\\tmkjn\\Documents\\python\\data_analysis\\data_analysis\\content\\audio_segments_1732030601\\segment_68.wav'},
{'text': 'それでも昔に比べたら', 'start': 210.33, 'end': 213.03, 'audio_path': 'C:\\Users\\tmkjn\\Documents\\python\\data_analysis\\data_analysis\\content\\audio_segments_1732030601\\segment_69.wav'},
{'text': '宇宙川の文化って大きくなったんだなと思う今日この頃ですよ', 'start': 213.306, 'end': 219.366, 'audio_path': 'C:\\Users\\tmkjn\\Documents\\python\\data_analysis\\data_analysis\\content\\audio_segments_1732030601\\segment_70.wav'},
{'text': '今が全盛期かな?', 'start': 220.026, 'end': 222.054, 'audio_path': 'C:\\Users\\tmkjn\\Documents\\python\\data_analysis\\data_analysis\\content\\audio_segments_1732030601\\segment_71.wav'},
{'text': 'ハハハハハ', 'start': 221.754, 'end': 222.95, 'audio_path': 'C:\\Users\\tmkjn\\Documents\\python\\data_analysis\\data_analysis\\content\\audio_segments_1732030601\\segment_72.wav'},
{'text': '毎年ね思ってるよ今年以上に盛り上がる年はないやろうにおうっていつも思ってるよ僕たちみんな', 'start': 223.77, 'end': 233.414, 'audio_path': 'C:\\Users\\tmkjn\\Documents\\python\\data_analysis\\data_analysis\\content\\audio_segments_1732030601\\segment_73.wav'},
{'text': 'こんなに楽しいこといっぱいさせてもらってさ今年が絶対全盛期', 'start': 234.266, 'end': 239.654, 'audio_path': 'C:\\Users\\tmkjn\\Documents\\python\\data_analysis\\data_analysis\\content\\audio_segments_1732030601\\segment_74.wav'},
{'text': 'だよねって', 'start': 239.802, 'end': 241.254, 'audio_path': 'C:\\Users\\tmkjn\\Documents\\python\\data_analysis\\data_analysis\\content\\audio_segments_1732030601\\segment_75.wav'},
{'text': 'でも次の年は次の年でなんか新しいこと起こったりしてさ', 'start': 241.178, 'end': 245.382, 'audio_path': 'C:\\Users\\tmkjn\\Documents\\python\\data_analysis\\data_analysis\\content\\audio_segments_1732030601\\segment_76.wav'},
{'text': '毎回それは思ってる', 'start': 245.626, 'end': 248.166, 'audio_path': 'C:\\Users\\tmkjn\\Documents\\python\\data_analysis\\data_analysis\\content\\audio_segments_1732030601\\segment_77.wav'},
{'text': 'だから毎年悔いのないように', 'start': 248.602, 'end': 251.366, 'audio_path': 'C:\\Users\\tmkjn\\Documents\\python\\data_analysis\\data_analysis\\content\\audio_segments_1732030601\\segment_78.wav'},
{'text': '活動頑張ろうって思ってんだけど', 'start': 251.29, 'end': 254.31, 'audio_path': 'C:\\Users\\tmkjn\\Documents\\python\\data_analysis\\data_analysis\\content\\audio_segments_1732030601\\segment_79.wav'},
{'text': 'よいしょ', 'start': 257.01800000000003, 'end': 258.63, 'audio_path': 'C:\\Users\\tmkjn\\Documents\\python\\data_analysis\\data_analysis\\content\\audio_segments_1732030601\\segment_80.wav'},
{'text': 'やめて', 'start': 258.714, 'end': 259.782, 'audio_path': 'C:\\Users\\tmkjn\\Documents\\python\\data_analysis\\data_analysis\\content\\audio_segments_1732030601\\segment_81.wav'}
]

source_segments = [{'text': 'ムラムラ', 'start': 0.0, 'end': 0.934, 'audio_path': 'C:\\Users\\tmkjn\\Documents\\python\\data_analysis\\data_analysis\\content\\audio_segments_1732030601\\segment_00.wav'},
{'text': 'ムラムラゲームラ?', 'start': 1.978, 'end': 2.854, 'audio_path': 'C:\\Users\\tmkjn\\Documents\\python\\data_analysis\\data_analysis\\content\\audio_segments_1732030601\\segment_01.wav'},
{'text': 'ゲージを取得する', 'start': 2.714, 'end': 4.134, 'audio_path': 'C:\\Users\\tmkjn\\Documents\\python\\data_analysis\\data_analysis\\content\\audio_segments_1732030601\\segment_02.wav'},
{'text': '相手に近い時避難時にゲージが少しずつたまるダメージ受けるとムラつくんだね確かにマリンドエムだからね', 'start': 4.186, 'end': 11.526, 'audio_path': 'C:\\Users\\tmkjn\\Documents\\python\\data_analysis\\data_analysis\\content\\audio_segments_1732030601\\segment_03.wav'},
{'text': 'ブラムラしちゃおっかな', 'start': 12.058, 'end': 14.118, 'audio_path': 'C:\\Users\\tmkjn\\Documents\\python\\data_analysis\\data_analysis\\content\\audio_segments_1732030601\\segment_04.wav'},
{'text': 'ムラムラゲージどこにある?', 'start': 14.97, 'end': 17.029999999999998, 'audio_path': 'C:\\Users\\tmkjn\\Documents\\python\\data_analysis\\data_analysis\\content\\audio_segments_1732030601\\segment_05.wav'},
{'text': '左下にあるわえやばいちょっとずつムラムラしてきた', 'start': 17.466, 'end': 21.19, 'audio_path': 'C:\\Users\\tmkjn\\Documents\\python\\data_analysis\\data_analysis\\content\\audio_segments_1732030601\\segment_06.wav'},
{'text': 'ちょっとずつ', 'start': 21.466, 'end': 22.982, 'audio_path': 'C:\\Users\\tmkjn\\Documents\\python\\data_analysis\\data_analysis\\content\\audio_segments_1732030601\\segment_07.wav'},
{'text': 'これあれじゃん敵の中に', 'start': 24.89, 'end': 27.206, 'audio_path': 'C:\\Users\\tmkjn\\Documents\\python\\data_analysis\\data_analysis\\content\\audio_segments_1732030601\\segment_08.wav'},
{'text': 'こんなに楽しいこといっぱいさせてもらってさ今年が絶対全盛期', 'start': 234.266, 'end': 239.654, 'audio_path': 'C:\\Users\\tmkjn\\Documents\\python\\data_analysis\\data_analysis\\content\\audio_segments_1732030601\\segment_74.wav'},
{'text': 'だよねって', 'start': 239.802, 'end': 241.254, 'audio_path': 'C:\\Users\\tmkjn\\Documents\\python\\data_analysis\\data_analysis\\content\\audio_segments_1732030601\\segment_75.wav'},
{'text': 'でも次の年は次の年でなんか新しいこと起こったりしてさ', 'start': 241.178, 'end': 245.382, 'audio_path': 'C:\\Users\\tmkjn\\Documents\\python\\data_analysis\\data_analysis\\content\\audio_segments_1732030601\\segment_76.wav'},
{'text': '毎回それは思ってる', 'start': 245.626, 'end': 248.166, 'audio_path': 'C:\\Users\\tmkjn\\Documents\\python\\data_analysis\\data_analysis\\content\\audio_segments_1732030601\\segment_77.wav'},
{'text': 'だから毎年悔いのないように', 'start': 248.602, 'end': 251.366, 'audio_path': 'C:\\Users\\tmkjn\\Documents\\python\\data_analysis\\data_analysis\\content\\audio_segments_1732030601\\segment_78.wav'},
{'text': '活動頑張ろうって思ってんだけど', 'start': 251.29, 'end': 254.31, 'audio_path': 'C:\\Users\\tmkjn\\Documents\\python\\data_analysis\\data_analysis\\content\\audio_segments_1732030601\\segment_79.wav'},
{'text': 'よいしょ', 'start': 257.01800000000003, 'end': 258.63, 'audio_path': 'C:\\Users\\tmkjn\\Documents\\python\\data_analysis\\data_analysis\\content\\audio_segments_1732030601\\segment_80.wav'},
{'text': 'やめて', 'start': 258.714, 'end': 259.782, 'audio_path': 'C:\\Users\\tmkjn\\Documents\\python\\data_analysis\\data_analysis\\content\\audio_segments_1732030601\\segment_81.wav'}
                   ]


print(compare_segments(clipping_segments, source_segments))
