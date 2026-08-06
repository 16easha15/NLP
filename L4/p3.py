#import lib
from nltk import word_tokenize
from nltk.corpus import stopwords
from string import punctuation
from pickle import load

#clean the data
sw = stopwords.words("english")
def clean_review(txt):
	txt = txt.lower()
	txt = word_tokenize(txt)
	txt =[t for t in txt if t not in punctuation]
	txt = [t for t in txt if t not in sw]
	txt = " ".join(txt)
	return txt

#model and cs back
with open("model.pkl","rb") as f:
	model = load(f)
	print("model ready")
	
with open("cv.pkl","rb") as f:
	cv = load(f)
	print("cv ready")

#prediction
re = input("enter review ")
cre = clean_review(re)
vre = cv.transform([cre])
ans = model.predict(vre)
if ans[0] == "p":
	print("wow we  made u happy")
else:
	print("sorry to let u down")

