import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
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

'''
tags
'''
tags["tag"] = tags["tag"].apply(lambda s : str(s).lower())
tags.isna().sum()
tags = tags.drop("timestamp", axis=1)
#print(tags.head(), "\nlength = ", len(tags))

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

#print(movies.isna().sum())
#print(len(movies))
    #13 in 9742 is negligeble
movies = movies.dropna()
#print(len(movies))

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
#print(movies.head())

'''
ratings
check outliers, no need for further preprocessing, standardisation and normalisation are not needed
'''

#print(set(ratings["rating"].tolist()))
testDist = [ratings["rating"].values]
sns.distplot(testDist, kde=False)
#plt.show()

'''
merge links with movies
'''
movies = pd.merge(movies,links, on="movieId", how="inner")

'''
export for further use
'''
movies.to_csv("exportedData/moviesExp.csv", index=False)
tags.to_csv("exportedData/tagsExp.csv", index=False)
ratings.to_csv("exportedData/ratingsExp.csv", index=False)

'''
DONE
'''