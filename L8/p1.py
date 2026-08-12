#import lib
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

#get the text from user
s1 = input("enter first sentence ")
s2 = input("enter second sentence ")

#combine the text
texts=[s1,s2]

#vectorize
cv = CountVectorizer()
vector = cv.fit_transform(texts)

#find the similarity
cs = cosine_similarity(vector)
print(cs)
ans = round(cs[0][1]*100,2)
print(ans)