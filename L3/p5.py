data = [10,40,30,22,23,45,56,76,81,92,93,45]
print(data)

edata = []
for d in data:
	if d % 2 ==0:
		edata.append(d)
print(edata)

nedata = [d for d in data if d % 2 == 0]
print(nedata)