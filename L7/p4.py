import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer

data = pd.read_csv("hotel_reviews.csv")
print(data)

#sentiment analysis
sia = SentimentIntensityAnalyzer()
def gps(txt):
	return sia.polarity_scores(txt)
data["ps"] = data["review"].apply(gps)
print(data)

def gs(txt):
	ps = sia.polarity_scores(txt)
	if ps["compound"]>=0.05:
		return "positive"
	elif ps["compound"] <= -0.05:
		return "negative"
	else:
		return "neutral"
	
data["sentiment"] = data["review"].apply(gs)
print(data)
data.to_csv("hresult.csv")
