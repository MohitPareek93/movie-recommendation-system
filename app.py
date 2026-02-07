import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

df = pd.read_csv('data/movies.csv')

df['overview'] = df['overview'].fillna('')

cv = CountVectorizer(stop_words='english')
count_matrix = cv.fit_transform(df['overview'])

cosine_sim = cosine_similarity(count_matrix, count_matrix)

def recommend(movie):
    idx = df[df['title'] == movie].index[0]
    scores = list(enumerate(cosine_sim[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)
    movies = [df['title'][i[0]] for i in scores[1:6]]
    return movies

print(recommend("Avatar"))
