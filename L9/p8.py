#import lib
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rapidfuzz import process

#load the data
data = pd.read_csv("movies.csv")
print(data)

#get the movie name
mn = input("enter movie name ")

#use rapidfuzz
res = process.extractOne(mn,data["movie_name"].tolist())
print(res)
mn = res[0]

#combined text
texts = [mn] + data["movie_name"].tolist()
print(texts)

#vectorize
cv = CountVectorizer()
vector = cv.fit_transform(texts)

#cs
cs = cosine_similarity(vector)
print(cs)

#score
score = cs[0][1:]
print(score)

#display
data["score"] = score * 100
print(data)

#sort & show
sdata = data.sort_values(by="score",ascending=False)
print(sdata)