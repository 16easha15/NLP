#import lib
import pytesseract
from PIL import Image


#get fn from user
fn = input("enter image file name ")

#some config
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe\tesseract.exe"

#get the text
stmt = pytesseract.image_to_string(Image.open(fn))

#show the text
print(stmt)