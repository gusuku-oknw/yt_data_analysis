import spacy

nlp = spacy.load('ja_ginza')

text = "これはテストですこれは2つ目の文です！最後の文です？"
doc = nlp(text)

for sent in doc.sents:
    print(sent.text)
