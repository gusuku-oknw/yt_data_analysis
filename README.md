以下がREADMEのドラフトです。

---

# **YouTube・Twitch チャットデータ解析プロジェクト**

## **概要**
本プロジェクトは、YouTubeやTwitchのライブチャットデータを収集・解析し、感情分析や可視化を行うシステムです。  
主な機能には、動画検索、チャットのダウンロード、音声解析、感情分析、字幕比較などがあります。

---

## **ディレクトリ構成**
```
.
├── christian-phu/                # BERTモデルの感情分析関連
├── fetch_sound/                   # 音声関連処理（精度調整中）
├── fetch_whisper/                 # Whisperを用いた音声認識
├── fetch_yt/                      # YouTube動画の検索・情報取得
├── sample_codes/                  # サンプルコード
├── train_chat/                    # チャット解析の学習データ
├── CSV_WordCloud.py               # チャットデータのワードクラウド生成
├── data_visualization.py          # データ可視化処理
├── data_visualization_bert-finetuned.py  # BERTを使用した可視化処理
├── line_distilbert.py             # 日本語感情分析用DistilBERTの実装
├── plot_chat_sentiment.py         # チャットの感情分析をプロット
├── plt_chat_emotions.py           # 解析結果を可視化
```

---

## **機能詳細**

### **1. 動画検索・取得**
#### `search_yt.py`
- YouTube Data API を用いて動画検索
- 指定したキーワードで動画を取得
- OpenAIのGPT-4oを活用し、切り抜き動画の概要欄から元動画URLを抽出

---

### **2. チャットダウンロード**
#### `chat_download.py`
- YouTubeおよびTwitchのチャットを取得
- チャットデータをCSV形式で保存
- スーパーチャット情報も取得可能

---

### **3. 音声データ処理**
#### `audio_utils.py`
- `download_yt_sound(url)`: YouTubeから音声をダウンロード
- `extract_vocals(audio_file)`: 音声ファイルからボーカルを抽出（Demucsを使用）

---

### **4. 感情分析**
#### `chat_emotions.py`
- `sentiment_pipeline()`: BERTを使用したポジティブ/ネガティブ/ニュートラル分析
- `weime_pipeline()`: Weimeを用いた感情分類（喜び・怒り・驚きなど）
- `mlask_pipeline()`: MLAskライブラリによる日本語感情分析

---

### **5. 字幕比較と音声認識**
#### `whisper_comparison.py`
- `transcribe_with_vad(audio_file)`: Silero VADで音声を分割し、Whisperで文字起こし
- `compare_segments(clipping_segments, source_segments)`: 切り抜き字幕と元動画字幕を比較し、一致する部分を抽出
- 類似度比較方法：
  - `sequence`: 文字列の類似度を比較
  - `jaccard`: 単語の共通度合いで類似度を測定
  - `tfidf`: TF-IDFを用いた類似度計算

---

## **セットアップ**
### **1. 必要なライブラリのインストール**
```bash
pip install -r requirements.txt
```

### **2. 環境変数の設定**
`.env`ファイルを作成し、YouTube APIキーとOpenAI APIキーを設定
```
YoutubeKey=YOUR_YOUTUBE_API_KEY
OpenAIKey=YOUR_OPENAI_API_KEY
```

---

## **使用方法**
### **1. YouTube動画検索**
```python
from search_yt import search_yt
sy = search_yt()
df = sy.search("にじさんじ 切り抜き")
df.to_csv("output.csv", index=False)
```

### **2. チャットダウンロード**
```python
from chat_download import chat_download
df = chat_download("https://www.youtube.com/watch?v=xxxxxxx")
df.to_csv("chat_data.csv", index=False)
```

### **3. 音声ダウンロード**
```python
from audio_utils import download_yt_sound
file_path = download_yt_sound("https://www.youtube.com/watch?v=xxxxxxx")
```

### **4. 感情分析**
```python
from chat_emotions import EmotionAnalyzer
analyzer = EmotionAnalyzer()
df = analyzer.analysis_emotion("chat_data.csv", ["sentiment", "weime", "mlask"])
```

### **5. 字幕比較**
```python
from whisper_comparison import WhisperComparison
wc = WhisperComparison()
source_segments = wc.transcribe_with_vad("source_audio.mp3")
clipping_segments = wc.transcribe_with_vad("clipping_audio.mp3")
matches, unmatched = wc.compare_segments(clipping_segments, source_segments)
```
