import MeCab
from mlask import MLAsk

emotion_analyzer = MLAsk()
text = "夕食がとても美味しく友達も喜んでいました。ありがとうございます！" \
       "客室担当方はフレンドリーで丁寧に接客してくれました。" \
       "朝ご飯もちょうどいいくらいの量で満足でした。" \
       "部屋も予想よりも広くびっくりしました。"
analyze_result = emotion_analyzer.analyze(text)
# print(analyze_result)
print(analyze_result["emotion"])
