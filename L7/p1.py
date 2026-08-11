#import lib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk import word_tokenize
from string import punctuation
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from pickle import dump

#load dataw
data = pd.read_csv("rest_reviews.tsv",sep="\t")
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

#feature and target
tv = TfidfVectorizer()
vector =tv.fit_transform(data["clean_review"])
features = pd.DataFrame(vector.toarray(),columns=tv.get_feature_names_out())
print(features)
target = data["Liked"]

#model
x_train,x_test,y_train,y_test = train_test_split(features.values,target)
model =MultinomialNB()
model.fit(x_train,y_train)

y_pred = model.predict(x_test)
cr = classification_report(y_test,y_pred)
print(cr)

with open("review_model.pkl","wb") as f:
	dump(model,f)
	print("model saved")

with open("review_tv.pkl","wb") as f:
	dump(tv,f)
	print("tv saved")



