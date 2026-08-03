#import lib
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer

#load data
data = pd.read_csv("housing.csv")
print(data)

#clean data
def split_info(data):
	data[["area","bedrooms"]]=data["info"].str.split(",",expand=True)
	data["area"]=data["area"].astype(float)
	data["bedrooms"]=data["bedrooms"].astype(float)
	return data[["area","bedrooms"]]

#features and target
features = data[["info"]]
target = data["price"]

print(features)
print(target) 

#model
model = make_pipeline(
	FunctionTransformer(split_info),
	LinearRegression()
	)
model.fit(features,target)

#prediction
area = float(input("enter area "))
bedrooms = float(input("enter number of bedrooms "))
test = pd.DataFrame({"info":[str(area)+","+str(bedrooms)]})
price = model.predict(test)
print(price[0])
