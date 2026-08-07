import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

data = pd.read_csv("movie_review.csv")
print(data)

tf = TfidfVectorizer()
vector = tf.fit_transform(data["review"])
res = pd.DataFrame(vector.toarray(),columns=tf.get_feature_names_out())
print(res)