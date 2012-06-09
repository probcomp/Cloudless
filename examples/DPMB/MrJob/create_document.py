from nltk.corpus import brown
for sentence in brown.sents():
    print " ".join(sentence)

# print " ".join(brown.words(brown.fileids()[0]))
