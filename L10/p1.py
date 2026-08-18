#import lib
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rapidfuzz import process

#load data
data = pd.read_csv("movies.csv")
print(data)

#get the movie genre
mg  = input("enter movie genre: ").lower()

#rapidfuzz
result = process.extractOne(mg,["animation","action","adventure","comedy","drama","thriller","fantasy","family","sci-fi","musical","crime"],score_cutoff=70)
print(result)
if result is None:
	print("no movies found")
	exit()
mg = result[0]
print(mg)

#combined texts
texts = [mg] + data["movie_genre"].str.lower().tolist()
print(texts)

#vectorize
cv = CountVectorizer()
vector = cv.fit_transform(texts)
print(vector)

#cs
cs = cosine_similarity(vector)
print(cs)

#score
score = cs[0][1: ]
data["score"]=score*100
print(data)

#show result
res = data.sort_values(by="score",ascending=False)
print(res)

fres = res[res["score"]>50]
if fres.shape[0]>0:
	ans = fres.sample(n=min(5,fres.shape[0]))
	print(ans)
else:
	print("no movie to suggest") 





