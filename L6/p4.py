#import lib
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

#load data
data = pd.read_csv("permission.csv")
print(data)

#check and handle null data
print(data.isnull().sum())

#features and target
cv = CountVectorizer()
vector = cv.fit_transform(data["text"])
features = pd.DataFrame(vector.toarray(),columns=cv.get_feature_names_out())
target = data["label"]
print(features)
print(target)

#model
model = MultinomialNB()
model.fit(features.values,target)

#prediction
text = input("enter text ")
vre = cv.transform([text])
ans = model.predict(vre)
print(ans)