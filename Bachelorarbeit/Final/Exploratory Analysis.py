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

#Ratings Distribution
temp = ratings["rating"]
temp.describe()
temp.median
temp.isnull().sum()
temp2 = ratings.groupby("userId").count().rating
temp2.sort_values(ascending=True)
temp2.describe()
sns.histplot(temp2)

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
ratings
check outliers, no need for further preprocessing, standardisation and normalisation are not needed
'''

#print(set(ratings["rating"].tolist()))
testDist = [ratings["rating"].values]
img = sns.displot(testDist, kde=False)
img
#plt.show()


'''
Exploratory Plots etc
'''
links.plot(kind="scatter", x="imdbId", y="tmdbId")
movies.plot(kind="scatter", x="movieId", y="Year")
sns.distplot(movies["Year"], kde=False)
links.head()

from timeit import default_timer
start = default_timer()
movies['Year'] = movies.title.str.extract("\((\d{4})\)", expand=True)
movies.Year = pd.to_datetime(movies.Year, format='%Y')
movies.Year = movies.Year.dt.year # As there are some NaN years, resulting type will be float (decimals)
movies.title = movies.title.str[:-7]
averageRating = ratings.groupby("movieId").mean().rating
averageRating
mov1 = pd.merge(movies, averageRating, on="movieId", how="inner")
mov1.head()
mov1.isnull().sum()
mov1 = mov1.sort_values(by="Year", ascending=True)
mov1.head()

import ssl
ssl._create_default_https_context = ssl._create_unverified_context
# Year distribution
sns.displot(mov1, x=mov1["Year"])
sns.displot(mov1, x=mov1["Year"], hue=mov1["Drama"])
sns.displot(mov1, x=mov1["Year"], hue=mov1["Action"])


movies.head()
temp = movies["Action"].value_counts()
temp[1]

# Genres Distribution
mov = pd.read_csv("originalData/movies.csv").copy()
mov.head()
genresOnehot = mov["genres"].str.get_dummies()
genresOnehot.head()
genresOnehot.sum()
temp = genresOnehot.sum()
temp.sort_values(ascending=False, inplace=True)
temp
sns.displot(temp)
img = temp.plot(kind="bar", title="Distribution of Genres")
img.set_ylabel("Count")
img

# Correlation between Year -- Average Rating?
averageRating = ratings.groupby("movieId").mean().rating
averageRating
mov = pd.merge(movies, averageRating, on="movieId", how="inner")
mov.head()
mov.isnull().sum()
mov = mov.sort_values(by="Year", ascending=True)

mov = pd.DataFrame({
    "Year" : mov["Year"],
    "Average Rating" : mov["rating"]
})

img = sns.displot(mov, x=mov["Year"], y=mov["Average Rating"])



#mov.to_csv("exportedData/yearRating.csv", index=False)
movies.head()
# Heatmap correlation of movie features
sns.heatmap(mov.corr(), annot=False)

ratings.head()
avgRproU = ratings.groupby("userId").mean().rating
avgRproU = avgRproU.rename("Average Movie Rating pro User")
sns.histplot(avgRproU, stat="density", kde=True, legend=True)
