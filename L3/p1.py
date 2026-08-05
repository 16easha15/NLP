import pandas as pd
import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
from string import punctuation
from sklearn.feature_extraction.text import CountVectorizer

nltk.download('stopwords')

data = pd.read_csv("car_data.csv")
print(data)

def clean_info(txt):
	txt = txt.lower()
	txt = word_tokenize(txt)
	txt = [t for t in txt if t not in stopwords.words("english")]
	txt = [t for t in txt if t not in punctuation]
	txt = " ".join(txt)
	return txt
data["clean_info"] = data["info"].apply(clean_info)
print(data)

cv=CountVectorizer()
vector = cv.fit_transform(data["info"])
print(vector)

res  = pd.DataFrame(vector.toarray(),columns=cv.get_feature_names_out())
print(res)