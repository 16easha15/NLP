#import lib
import pandas as pd
from nltk import word_tokenize
from nltk.corpus import stopwords
from string import punctuation
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from pickle import dump

#load the data
data = pd.read_csv("movie_review_2.csv")
print(data)

#clean the data
sw = stopwords.words("english")
def clean_review(txt):
	txt = txt.lower()
	txt = word_tokenize(txt)	
	txt = [t for t in txt if t not in punctuation]
	txt = [t for t in txt if t not in sw]
	txt = " ".join(txt)
	return txt
data["clean_review"]=data["review"].apply(clean_review)
print(data)

#features and target
cv = CountVectorizer()
vector = cv.fit_transform(data["clean_review"])
features = pd.DataFrame(vector.toarray(),columns=cv.get_feature_names_out())
target = data["result"]


print(features)
print(target)

#model
model =MultinomialNB()
model.fit(features.values,target)

#model and cv save
with open("model.pkl","wb") as f:
	dump(model,f)
	print("model saved")

with open("cv.pkl","wb") as f:
	dump(cv,f)
	print("cv saved")

