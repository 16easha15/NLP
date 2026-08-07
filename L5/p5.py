#import lib
import pandas as pd
from pickle import load 
from nltk import word_tokenize
from nltk.corpus import stopwords
from string import punctuation
from nltk.stem import PorterStemmer

#data cleaning 
sw = stopwords.words("english")
ps = PorterStemmer()
def clean_text(txt):
	txt = txt.lower()
	txt = word_tokenize(txt)
	txt =[t for t in txt if t not in punctuation]
	txt =[t for t in txt if t not in sw]
	txt =[ps.stem(t) for t in txt]
	txt =" ".join(txt)
	return txt

# get back model and tf
with open("news_model.pkl","rb") as f:
	model = load(f)
	print("model ready")

with open("news_tf.pkl","rb") as f:
	tf = load(f)
	print("tf ready")

#predictionw
text = input("enter text ")
ctext = clean_text(text)
vtext = tf.transform([ctext])
ans = model.predict(vtext)
print(ans)