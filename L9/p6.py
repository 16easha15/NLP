from rapidfuzz import fuzz

s1 = input("enter first stmt ")
s2 = input("enter second stmt ")

res = fuzz.token_sort_ratio(s1,s2)
print(res)