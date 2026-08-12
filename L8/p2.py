#import lib
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

#get the text from from file
fn1 = input("enter first file name ")
with open(fn1,"r") as f1:
	s1 = f1.read()
fn2 = input("enter second file name ")
with open(fn2,"r") as f2:
	s2 = f2.read()

#combine the text
texts = [s1,s2]

#vectorize
cv = CountVectorizer()
vector = cv.fit_transform(texts)

#find the similarity
cs = cosine_similarity(vector)
print(cs)
ans = round(cs[0][1]*100,2)
print(ans)