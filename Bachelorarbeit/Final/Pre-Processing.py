import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
plt.style.use("ggplot")
pd.options.display.max_columns = None
pd.options.display.width=None
import re

'''
read
'''
tags = pd.read_csv("originalData/tags.csv")
movies = pd.read_csv("originalData/movies.csv")
links = pd.read_csv("originalData/links.csv")
ratings = pd.read_csv("originalData/ratings.csv")

tags.head()
ratings.head()
movies.head()
links.head()

'''
tags (not used further in the Thesis)
'''
tags["tag"] = tags["tag"].apply(lambda s : str(s).lower())
tags.isna().sum()
tags = tags.drop("timestamp", axis=1)

'''
movies
'''
def extractYear():

    reg = ".*\((\d+)\)"
    counter = 0
    while counter < len(movies):
        res = re.findall(reg, movies["title"][counter])
        counter += 1

    def get_movie_year(title):
            reg = ".*\((\d+)\)"
            res = re.findall(reg, str(title))
            if len(res) != 1:
                return np.nan
            else:
                return int(res[0])

    movies["Year"] = movies.title.apply(get_movie_year)
extractYear()

movies.isna().sum()
#13 in 9742 is negligeble
movies = movies.dropna()

'''
split, oneHot, addList
'''
movies["genres_list"] = movies.genres.apply(lambda s : s.split("|"))
genres = set()
def foo(l):
    global genres
    for e in l:
        genres.add(str(e))
    return
movies["genres_list"].apply(foo)
def boolean_df(item_lists, unique_items):
    # Create empty dict
    bool_dict = {}

    # Loop through all the tags
    for i, item in enumerate(unique_items):
        # Apply boolean mask
        bool_dict[item] = item_lists.apply(lambda x: item in x)

    # Return the results as a dataframe
    return pd.DataFrame(bool_dict)
expanded_genres = boolean_df(movies["genres_list"], genres)
movies = pd.concat([movies, expanded_genres], axis = 1)
movies = movies.drop("genres", axis=1)


'''
merge links with movies
'''
movies = pd.merge(movies,links, on="movieId", how="inner")


'''
Export Uncut versions for CBF and Top-N Recommender, tags is not needed.
'''
movies.to_csv("exportedData/moviesExpUncut.csv", index=False)
ratings.to_csv("exportedData/ratingsExpUncut.csv", index=False)


'''
Cut movies with rating count<0.9 Percentile of movie ratings count ==> 27 ratings
'''
ratings = ratings.copy()
ratings["ratings_count"] = ratings.groupby("movieId")["rating"].count().sort_values(ascending=False)
ratings = ratings.reset_index()

#0.9 percentile
s = np.array(ratings["ratings_count"])
perc90 = np.percentile(s, 90)

ratings = ratings[ratings["ratings_count"] > perc90]
movies = movies.merge(ratings, on="movieId", how="inner") #check if correct!!


'''
Export Cut Versions
'''
movies.to_csv("exportedData/moviesExpCut.csv", index=False)
ratings.to_csv("exportedData/ratingsExpCut.csv", index=False)


