import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

txt = ["food was not good but staff was courteous"]
cv = CountVectorizer(ngram_range=(1,3))
vector = cv.fit_transform(txt)
print(vector)

res = pd.DataFrame(vector.toarray(),columns=cv.get_feature_names_out())
print(res)

#unigram (1,1)
#bigram (2,2)
#trigram (3,3)
#ug +bg (1,2)
#bg + tg (2,3)
#u + b+t (1,3)