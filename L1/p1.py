
#Tokenization

#ssv data (space seperated values)
names = "amit sumit neha pooja"
res = names.split(" ")
for r in res:
	print(r)

#csv data (comma seperated values)
names = "elon,jeff,steve,mark"
res = names.split(",")
for r in res:
	print(r)

#hsv data (hash seperated values)
numbers ="10#30#40#20#40"
res = numbers.split('#')
sum = 0
for r in res:
	sum = sum + int(r)
print(sum)