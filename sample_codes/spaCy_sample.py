import spacy

nlp = spacy.load("ja_core_news_trf")  # GPU対応のトランスフォーマーモデル
doc = nlp("ムラムラゲージを取得する相手に近い時避難時にゲージが少しずつたまるダメージ受けるとムラムラつくんだね確かにマリンドエムだからねムラムラしちゃうかなモザイクかってたちょっとずつこれあれじゃんこれあれじゃん敵の中にマザイクかかってた今は面白すぎるでしょちょっとちょっとするんだなおいしょこれかじしの道具来たねこれちょっと取りたいこれ金床2回使えるようになるから結構お得なんだよねおいしょおいしょああむらぶらしてきましたこのホントビスインのこの体をめちゃくちゃにしてくださいだねシルを飛ばしまくっているよ今のマリンが僕のマリンがマリンとワミイツイのコラボレーションやばいのでは超汁飛ばしまくってるけどシルっ気が大変なことなってますよすごい!シーンの姿来た!かわいい懐かしいレビュー配信でココちゃんがさ絵本絵本調で動画つけるの?これもヒーロイドアクションの一つだろうと倒した相手がランダムでいろいろ落とす何いろいろって触れるとそれを持ち上げ相手に投げつけるのこれもヒートアクションの一つだろうこれわ怖いびっくりゃうわ怖い!50キロの握力ギュッキュッキュッキュッ怖い怖い怖いめっちゃ怖いやがー50キロの握力怖かったんだ怖っ怖すぎですよ50キロの悪力怖い怖い怖い怖いめちゃ怖いんやがーあかーうわー!ヒーユだと思ってたらマジのゴリランの手出てきてるコシヨミたちの群れがあれかな昨日のライブのライブの帰り道だったかもしれない昨日のライブめっちゃよかったわルイと一緒に家で見てたんだけどさあるいんちでやばいね僕が入った時はVTuberの文化はまだまだこれから発展していくぞって感じだったけど今もそうなんだけどさそれでも昔に比べたらVTuberの文化って大きくなったんだなと思う今日この頃ですよ今年以上に盛り上がる年はないやろにおおおうっていつも思ってるよ僕たちみんなこんなに楽しいこといっぱりしてさ毎回それは思ってるだから毎年苦意のないように活動頑張ろうって思ってんだけどやばい死ぬ死ぬやめてちゃやめてちゃやめてちゃやめてちゃやめてちゃめて")
sentences = [sent.text for sent in doc.sents]

for i, sent in enumerate(sentences):
    print(f"文{i + 1}: {sent}")
    print()
