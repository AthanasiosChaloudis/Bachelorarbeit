#main libraries
import numpy as np
import pandas as pd
pd.options.display.max_columns = None
pd.options.display.width=None

#tf idf sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
#from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
#from sklearn.metrics.pairwise import linear_kernel

#tfidf tools
from ast import literal_eval

movies = pd.read_csv("exportedData/moviesExp.csv", usecols=["movieId","title", "tmdbId", "genres_list"])
metadata = pd.read_csv("originalData/movies_metadata.csv", usecols=["id", "overview", "vote_average", "vote_count"])
credits = pd.read_csv("originalData/credits.csv")
keywords = pd.read_csv("originalData/keywords.csv")

credits["id"] = credits["id"].astype("int")
keywords["id"] = keywords["id"].astype("int")
metadata["id"] = metadata["id"].astype("int")

moviesExt = pd.merge(credits, keywords, on="id", how="inner")
moviesExt = pd.merge(moviesExt, metadata, on="id", how="inner")
moviesExt = moviesExt.drop_duplicates()

moviesExt.isna().sum()
moviesExt = moviesExt.dropna()
moviesExt = moviesExt.merge(movies, left_on="id", right_on="tmdbId", how="inner")

for x in range(len(moviesExt)):
    moviesExt["genres_list"][x] = moviesExt["genres_list"][x].strip('][').split(', ')

features = ["cast", "crew", "keywords"]
for feature in features:
    moviesExt[feature] = moviesExt[feature].apply(literal_eval)

def get_director(x):
    for i in x:
        if i["job"] == "Director":
            return i["name"]
    return np.nan

def get_list(x):
    if isinstance(x, list):
        names = [i["name"] for i in x]
        #Check if more than 3 elements exist. If yes, return only first three. If no, return entire list.
        if len(names) > 3:
            names = names[:3]
        return names

    #Return empty list in case of missing/malformed data
    return []

def str_cleanUp(x):
    if isinstance(x, list):
        return [str.lower(i.replace(" ", "")) for i in x]
    else:
        #Check if director exists. If not, return empty string
        if isinstance(x, str):
            return str.lower(x.replace(" ", ""))
        else:
            return ''

# Define new director, cast, genres and keywords features that are in a suitable form.
moviesExt["director"] = moviesExt["crew"].apply(get_director)

features = ["cast", "keywords"]
for feature in features:
    moviesExt[feature] = moviesExt[feature].apply(get_list)

features = ["cast", "keywords", "director", "genres_list"]
for feature in features:
    moviesExt[feature] = moviesExt[feature].apply(str_cleanUp)

for i in range(len(moviesExt)):
    for j in range(len(moviesExt["genres_list"][i])):
        moviesExt["genres_list"][i][j] = literal_eval(moviesExt["genres_list"][i][j])

moviesExt.isnull().sum()


'''
UNTIL NOW PREPROCESSING, FILTERING NOW
'''

contentBasedRecom = moviesExt.copy()
def featureJoin(x):
    return ' '.join(x['keywords']) + ' ' + ' '.join(x['cast']) + ' ' + x['director'] + ' ' + ' '.join(x['genres_list'])
contentBasedRecom['soup'] = contentBasedRecom.apply(featureJoin, axis=1)

'''
TF - IDF Vector Space Model
'''

tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(contentBasedRecom['soup'])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

indices = pd.Series(contentBasedRecom.index, index=contentBasedRecom['title']).drop_duplicates()

def content_Recommendations(title):
    idx = indices[title]

    # Get the pairwsie similarity scores of all movies with that movie
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the movies based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the 10 most similar movies
    sim_scores = sim_scores[1:11]

    # Get the movie indices
    movie_indices = [i[0] for i in sim_scores]

    # Return the top 10 most similar movies
    return contentBasedRecom['title'].iloc[movie_indices], sim_scores

result = pd.DataFrame()
def recommend(title):
    result["title"] = content_Recommendations(title)[0]
    temp = np.array(content_Recommendations(title)[1])
    result["cos_sim"] = temp[:, 1]
    print(result)

recommend('Dark Knight, The (2008)')

#result.plot.barh(x="title")

