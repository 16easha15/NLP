#pip install PyMuPDF

#import lib
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pymupdf

#get the text from file
fn1 = input("enter first file name ")
f1 = pymupdf.open(fn1)
s1 = ""
for f in f1:
	s1 = s1 + f.get_text()
f1.close()

fn2 = input("enter second file name ")
f2 = pymupdf.open(fn2)
s2 = ""
for f in f2:
	s2 = s2 +f.get_text()
f2.close()

#combine the text
texts = [s1,s2]

#vectorixe
cv = CountVectorizer()
vector = cv.fit_transform(texts)

#find the similarity
cs = cosine_similarity(vector)
print(cs)
ans = round(cs[0][1]*100,2)
print(ans)