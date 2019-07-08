import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from rake_nltk import Rake


data = pd.read_csv('./dataframe/IMDB-Movie-Data/IMDB-Movie-Data.csv')
data = data[['Title','Genre','Description']]

data['tags']= ""

for index, row in data.iterrows():
    desc_keywords = row['Description']
    r = Rake()
    r.extract_keywords_from_text(desc_keywords)
    key_words_dict_scores = r.get_word_degrees()
    row['tags'] = list(key_words_dict_scores.keys())

data.drop(columns = ['Description'], inplace = True)
data.set_index('Title', inplace = True)

data['similarity'] = ''
columns = data.columns
for index, row in data.iterrows():
    words = ''
    for col in columns:
            words = words + str(row[col]) + ' '
    row['similarity'] = words


data.drop(columns=[col for col in data.columns if col != 'similarity'], inplace=True)



cv = CountVectorizer()
count_matrix = cv.fit_transform(data['similarity'])


cos_sim = cosine_similarity(count_matrix, count_matrix)

movie = input("What movie have you watched? : ")

#movie = "Inception"

recommended_movies = []

indices = pd.Series(data.index)
idx = indices[indices == movie].index[0]
score = pd.Series(cos_sim[idx]).sort_values(ascending=False)
top_10 = list(score.iloc[1:26].index)
for i in top_10:
    recommended_movies.append(list(data.index)[i])

print("TOP 25 similar movies with " + movie +" are")
print(recommended_movies)