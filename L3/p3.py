from nltk.corpus import stopwords

res = stopwords.words("english")
print(res)
print(len(res))

res.remove("about")
print(res)
print(len(res))

