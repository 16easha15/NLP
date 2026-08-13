from rapidfuzz import process

names = ["kamal","vimal","amit","namit"]
name = input("enter name ")

res = process.extractOne(name,names)
print(res)

res = process.extract(name,names)
print(res)