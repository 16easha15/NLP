#import lib
import pandas as pd
from nltk import word_tokenize
from nltk.corpus import stopwords
from string import punctuation
from nltk.stem import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report

#load the data
data = pd.read_csv("spam.csv")
print(data)

#check and handle null data
print(data.isnull().sum())

#data cleaning

sw = stopwords.words("english")
ps = PorterStemmer()
def clean_text(txt):
	txt=txt.lower()
	txt = word_tokenize(txt)
	txt = [t for t in txt if t not in punctuation]
	txt = [t for t in txt if t not in sw]
	txt = [ps.stem(t) for t in txt]	
	txt = " ".join(txt)
	return txt

data["clean_txt"] = data["text"].apply(clean_text)
print(data)

#features and target
cv =CountVectorizer()
vector = cv.fit_transform(data["clean_txt"])
features = pd.DataFrame(vector.toarray(),columns=cv.get_feature_names_out())
columns = cv.get_feature_names_out()
print(features)
target = data["label"]
print(target)

#train and test
x_train,x_test,y_train,y_test = train_test_split(features.values,target)


#model
model = MultinomialNB()
model.fit(x_train,y_train)

#classification_report
y_pred = model.predict(x_test)
cr = classification_report(y_test,y_pred)
print(cr)

#prediction
text = input("enter message ")
ctext = clean_text(text)
vtext = cv.transform([ctext])
ans = model.predict(vtext)
print(ans)
