#import lib
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import make_pipeline

#load the data
data = pd.read_csv("area_location_price.csv")
print(data)

#clean data
def clean_info(data):
	data["area"] = data["info"].str.extract(r'area=(\d+)')
	data["area"] = data["area"].astype(float)
	data["location"]=data["info"].str.extract(r'location=(\w+)')
	data =pd.get_dummies(data,columns=["location"],dtype=int)
	if "location_karjat" not in data.columns:
		data["location_karjat"]=0
	if "location_khandala" not in data.columns:
		data["location_khandala"]=0
	if "location_lonavala" not in data.columns:
		data["location_lonavala"]=0
	return data[["area","location_karjat","location_khandala","location_lonavala"]]

#features and target
features = data[["info"]]
print(features)
target = data["price"]

#model
model = make_pipeline(
	FunctionTransformer(clean_info),
	LinearRegression()
	)
model.fit(features,target)

#prediction
area = int(input("enter area in square feet "))
location = input("enter location:karjat/khandala/lonavala ").lower()
test = pd.DataFrame({
	"info":["area=" + str(area) + " location=" +str(location)]
})
price = model.predict(test)
print(price)