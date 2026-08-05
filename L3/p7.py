#import lib
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

#load the data
data = pd.read_csv("movie_review_1.csv")
print(data)

#clean the data

#vectorize
cv = CountVectorizer()
vector = cv.fit_transform(data["review"])

#features and target
features = pd.DataFrame(vector.toarray(),columns=cv.get_feature_names_out())
target = data["result"]
print(features)
print(target)

#model
model = MultinomialNB()
model.fit(features.values,target)

#prediction
re = input("enter movie review ")
vre = cv.transform([re])
ans = model.predict(vre)
print(ans)
if ans[0]=='p':
	print("wow we are thrilled")
else:
	print("oh sorry for that exp")

#internal
print(model.predict_proba(vre))