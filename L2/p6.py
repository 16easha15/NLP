#import lib
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import FunctionTransformer,OneHotEncoder
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer

#load data
data = pd.read_csv("area_location_price.csv")
print(data)

#clean data
def clean_info(data):
	data["area"] = data["info"].str.extract(r'area=(\d+)')
	data["location"] = data["info"].str.extract(r'location=(\w+)')
	return data[["area","location"]]

#features and target
features = data[["info"]]
target = data["price"]


#model
model = make_pipeline(
	FunctionTransformer(clean_info),
	ColumnTransformer(transformers = [("location",OneHotEncoder(handle_unknown="ignore"),["location"]) ]),
	LinearRegression()
	)
model.fit(features,target)

#prediction
area = int(input("enter area in square feet "))
location = input("enter location:karjat/khandala/lonavala ").lower()
test = pd.DataFrame({"info":["area=" +str(area)+ "location=" +str(location)]})
price = model.predict(test)
print(price)