#import lib
import pandas as pd
from sklearn.linear_model import LinearRegression

#load the data
data = pd.read_csv("housing2.csv")
print(data)

#clean data
def split_info(data):
	data[["area","bedrooms"]]=data["info"].str.split(" ",expand=True)
	data["area"]=data["area"].str.replace("area=","").astype(float)
	data["bedrooms"]=data["bedrooms"].str.replace("bedrooms=","").astype(float)
	return data[["area","bedrooms"]]

#features and targets
features = split_info(data)
target = data["price"]
print(features)
print(target)

#model
model = LinearRegression()
model.fit(features,target)

#prediction
area = float(input("enter area "))
bedrooms = float(input("enter number of bedrooms "))
test = pd.DataFrame({"info":[str(area)+" "+str(bedrooms)]})
price = model.predict(split_info(test))
print(price[0])