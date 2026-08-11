#adv - it is pretrained model
#conside

from nltk.sentiment.vader import SentimentIntensityAnalyzer

sia = SentimentIntensityAnalyzer()	#pos
s1 = "easha is a good girl"
print(sia.polarity_scores(s1))

s1 = "easha is a good girl !!!!"	#emoticons
print(sia.polarity_scores(s1))

s1 = "easha is a good girl but naughty at times"	#conjunctions
print(sia.polarity_scores(s1))

s1 = "easha is a bad girl"		#negative
print(sia.polarity_scores(s1))

s1 = "easha is a BAD girl"		#change case
print(sia.polarity_scores(s1))

s1 = "easha is a very bad girl"		#degree modifiers
print(sia.polarity_scores(s1))

s1 = "easha is a dangerous girl hurrible girl"
print(sia.polarity_scores(s1))