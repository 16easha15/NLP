from flask import *
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pymupdf

app = Flask(__name__)

@app.route("/",methods=["POST","GET"])
def home():
	if request.method == "POST":
		try:
			f1 = request.files["f1"]
			f2 = request.files["f2"]
		
			d1 = pymupdf.open(stream=f1.read(),filetype="pdf")
			s1 = ""
			for d in d1:
				s1 = s1 + d.get_text()
			d1.close()
				
			d2 = pymupdf.open(stream=f2.read(),filetype="pdf")
			s2 = ""
			for d in d2:
				s2 = s2 + d.get_text()
			d2.close()
		except pymupdf.FileDataError:
			msg = "invalid file format"
			return render_template("home.html",msg=msg)
		#combine the text
		texts = [s1,s2]
	
		#vectorize
		cv = CountVectorizer()
		vector = cv.fit_transform(texts) 
	
		#find the similarity
		cs = cosine_similarity(vector)
		print(cs)
		ans= round(cs[0][1] *100,2)
		msg = "Similarity = "+str(ans)
		return render_template("home.html",msg=msg)
	else:
		return render_template("home.html")

if __name__ == "__main__":
	app.run(debug=True,use_reloader =True)