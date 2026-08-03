#Tokenization

from nltk import word_tokenize

stmt = "I love My India!!!"
res = word_tokenize(stmt)
for r in res:
	print(r)

stmt = "Hey Der!How are You???"
res = word_tokenize(stmt)
for r in res:
	print(r)

names = "elon,jeff,mark,sundar"
res = word_tokenize(names)
for r in res:
	print(r)