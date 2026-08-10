#import lib
import pandas as pd
from pickle import load
from nltk import word_tokenize
from string import punctuation
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

#load the data
data = pd.read_csv("news_test.csv")
print(data)

#check and remove null data
print(data.isnull().sum())

#clean the data
sw = stopwords.words("english")
sw.remove("not")
sw.remove("don't")
ps = PorterStemmer()
def clean_text(txt):
	txt = txt.lower()
	txt = word_tokenize(txt)	
	txt = [t for t in txt if t not in punctuation]
	txt = [t for t in txt if t not in sw]
	txt = [ps.stem(t) for t in txt]
	txt = " ".join(txt)
	return txt
data["clean_text"] = data["text"].apply(clean_text)

#restore model and tv
with open("news_model.pkl","rb") as f:
	model = load(f)
	print("model back")

with open("news_tv.pkl","rb") as f:
	tv = load(f)
	print("tv back")

#prediction
x_data = data["text"]
cx_data = data["text"].apply(clean_text)
vx_data = tv.transform(cx_data)
y_pred = model.predict(vx_data)
data["result"] = y_pred.tolist()
data.to_csv("news_test.csv")

