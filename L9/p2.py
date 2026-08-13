#import lib
import pytesseract
from PIL import Image
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

#some config
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe\tesseract.exe"

#get fn from user
f1 = input("enter image file name 1 ")
s1 = pytesseract.image_to_string(Image.open(f1))
f2 = input("enter image file name 2 ")
s2 = pytesseract.image_to_string(Image.open(f2))

#combine text
texts = [s1,s2]

#vectorize
cv = CountVectorizer()
vector = cv.fit_transform(texts)

#find similarity
cs = cosine_similarity(vector)
ans = round(cs[0][1]*100,2)
print(ans,"%")