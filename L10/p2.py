#import lib
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

#Load the data
data = pd.read_csv("ek.csv")
#print(data)

#print("Welcome to kamal classes,i am kobot,press e to exit")
while True:
	qts= input("enter qts --> ")
	if qts == "e":
		print("kobot --> bye ")
		exit()
	else:
		#texts
		texts = [qts]+data["question"].str.lower().tolist()
		
		#vectorize
		cv = CountVectorizer()
		vector = cv.fit_transform(texts)

		#cs
		cs = cosine_similarity(vector)	
		#score
		score = cs[0][1: ]
		data["score"] = score *100
		#print(data)
		
		#reply
		res = data.sort_values(by="score",ascending=False)
		fres = res[res["score"]>0]
		#print(res)

		if fres.shape[0]>0:
			print("kobot-->",fres["answer"].values[0])	
		else:
			print("kobot-->sorry i dont know")
		
		