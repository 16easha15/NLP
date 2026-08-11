#import lib
import pandas as pd
from nltk import word_tokenize
from string import punctuation
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from pickle import load

#load dataw
data = pd.read_csv("new_reviews.tsv",sep="\t")
print(data)

#check and handle null data
print(data.isnull().sum())

#cleaning data
sw = set(stopwords.words("english"))
sw.remove("not")
sw.remove("don't")
ps = PorterStemmer()
def clean_review(txt):
	txt = txt.lower()
	txt = word_tokenize(txt)
	txt = [t for t in txt if t not in punctuation]	
	txt = [t for t in txt if t not in sw]
	txt = [ps.stem(t) for t in txt]
	txt = " ".join(txt)
	return txt
	
data["clean_review"]=data["Review"].apply(clean_review)
print(data)

#model

with open("review_model.pkl","rb") as f:
	model = load(f)
	print("model is back ")

with open("review_tv.pkl","rb") as f:
	tv = load(f)
	print("tv is back")

#prediction
x_data = data["Review"]
cx_data = data["Review"].apply(clean_review)
vx_data = tv.transform(cx_data)
y_pred = model.predict(vx_data)
data["result"] = y_pred.tolist()
#data.to_csv("result.csv")
data.to_csv("result.tsv",sep="\t")
print("done")

