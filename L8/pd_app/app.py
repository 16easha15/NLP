from flask import *
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

@app.route("/",methods=["POST","GET"])
def home():
	if request.method == "POST":
		f1 = request.files["f1"]
		f2 = request.files["f2"]

		s1 = f1.read().decode("utf-8")
		s2 = f2.read().decode("utf-8")
	

		#combine the text
		texts = [s1,s2]
	
		#vectorize
		cv = CountVectorizer()
		vector = cv.fit_transform(texts)
	
		#find the similarity
		cs = cosine_similarity(vector)
		print(cs)
		ans =round(cs[0][1]*100,2)
		msg = "Similarity = "+str(ans)
		return render_template("home.html",msg=msg)
	
	else:
		return render_template("home.html")

if __name__ == "__main__":
	app.run(debug=True,use_reloader=True)

