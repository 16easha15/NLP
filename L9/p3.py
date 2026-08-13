#import lib
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

#load the data
data = pd.read_csv("resumes.csv")
print(data)

#get the jd -> job description
jd = input("enter job description ")

#combined text
texts = [jd] + data["tech_skills"].tolist()
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