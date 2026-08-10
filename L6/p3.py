#import lib
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

#load data
data = pd.read_csv("movie_reviews.csv")
print(data)

#check and handle null data
print(data.isnull().sum())

#features and target
cv = CountVectorizer(ngram_range = (2,2))
vector = cv.fit_transform(data["review"])
features = pd.DataFrame(vector.toarray(),columns=cv.get_feature_names_out())
target = data["result"]
print(features)
print(target)

#model
model = MultinomialNB()
model.fit(features.values,target)

#prediction
re = input("enter review ")
vre = cv.transform([re])
ans = model.predict(vre)
print(ans)