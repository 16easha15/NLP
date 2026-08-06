import pandas as pd
from nltk import word_tokenize
from nltk.corpus import stopwords
from string import punctuation
from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer

data = pd.read_csv("sld2.csv")
print(data)

sw = stopwords.words("english")
ss = SnowballStemmer("english")
def clean_review(txt):
	txt = txt.lower()
	txt = word_tokenize(txt)
	txt = [t for t in txt if t not in punctuation]
	txt = [t for t in txt if t not in sw]
	txt = [ss.stem(t) for t in txt]
	txt = " ".join(txt)
	return txt
	
data["clean_review"] = data["review"].apply(clean_review)
print(data)

cv = CountVectorizer()
vector = cv.fit_transform(data["clean_review"])
print(vector)

features = pd.DataFrame(vector.toarray(),columns=cv.get_feature_names_out())
print(features)