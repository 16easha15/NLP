#import lib
import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.metrics  import classification_report


#load data
data = pd.read_csv("food_reviews.csv")
print(data)


#sentiment anlysis
sia = SentimentIntensityAnalyzer()

def gps(text):
	return sia.polarity_scores(text)

data["ps"] = data["text"].apply(gps)
print(data)

def gs(text):
	ps = sia.polarity_scores(text)
	if ps["compound"]>=0.05:
		return "positive"
	elif ps["compound"] <= -0.05:
		return "negative"
	else:
		return "neutral"


data["nlabel"]=data["text"].apply(gs)
print(data)

cr = classification_report(data["label"],data["nlabel"],zero_division=0)
print(cr)
