from flask import *
from nltk import word_tokenize
from nltk.corpus import stopwords
from string import punctuation
from pickle import load

#clean the data
sw = stopwords.words("english")
def clean_review(txt):
	txt = txt.lower()
	txt = word_tokenize(txt)
	txt = [t for t in txt if t not in punctuation]
	txt = [t for t in txt if t not in sw]
	txt = " ".join(txt)
	return txt

#model and cv back
with open("model.pkl","rb") as f:
	model =load(f)
	print("model ready ")

with open("cv.pkl","rb") as f:
	cv =load(f)
	print("cv ready ")

app = Flask(__name__)
@app.route("/",methods=["POST","GET"])
def home():
	if request.method == "POST":
		re =  request.form.get("review")
		cre = clean_review(re)
		vre = cv.transform([cre])
		ans = model.predict(vre)
		if ans[0] == "p":
			msg ="wow we made u happy"	
		else:
			msg = "Sorry to let u down"
		return render_template("home.html",msg=msg)
	else:
		return render_template("home.html")
if __name__=="__main__":
	app.run(debug=True,use_reloader=True)