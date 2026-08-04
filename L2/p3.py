import pandas as pd

data = pd.DataFrame({
	"info":[
		"name=kamal mobile=8378456734 college=rait",
		"name=easha mobile=9045672238 college=scoe",
		"name=tanmay mobile=7856345678 college=dyp",
		"name=alok mobile=9067458723 college=ltcoe",
	]
})
print(data)
data["name"]=data["info"].str.extract(r'name=(\w+)')
data["college"]=data["info"].str.extract(r'college=(\w+)')
data["mobile"]=data["info"].str.extract(r'mobile=(\d+)')
print(data)