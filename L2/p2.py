#import lib
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import make_pipeline

#load the data
data = pd.read_csv("cars.csv")
print(data)

#clean the data
def split_info(data):
	data[["engine","enginecap","fuel","fueltype"]]=data["details"].str.split(" ",expand=True)
	data["enginecap"] = data["enginecap"].str.replace("cc","").astype(float)
	data = pd.get_dummies(data,columns=["fueltype"],dtype=int)
	if "fueltype_petrol" not in data.columns:
		data["fueltype_petrol"]=0
	if "fueltype_diesel" not in data.columns:
		data["fueltype_diesel"]=0
	return data[["enginecap","fueltype_diesel","fueltype_petrol"]]

#features and targets
features = data[["details"]]
print(features)
target = data["price"]
print(target)

#model
model = make_pipeline(
	FunctionTransformer(split_info),
	LinearRegression()
	)
model.fit(features,target)

#prediction
cap =int(input("enter engine cap "))
fuel = input("enter fuel type ").lower()
test = pd.DataFrame({
	"details": ["engine " + str(cap) + "cc fuel " + str(fuel)]
})
price = model.predict(test)
print(price)
