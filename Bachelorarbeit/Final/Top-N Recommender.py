#main libraries
import numpy as np
import pandas as pd
pd.options.display.max_columns = None
pd.options.display.width=None

movies = pd.read_csv("exportedData/moviesExpCut.csv", usecols=["movieId, title"]).copy()
ratings = pd.read_csv("exportedData/ratingsExpCut.csv", usecols=["movieId, rating"]).copy()


#NEEDS A LOT OF REWORK, MAKE MORE COMPREHENSIVE, CHANGE NAMES, REDUCE VARIABLES AMOUNT

'''
find most rated movies
'''
moviesRated = pd.DataFrame(data=pd.merge(movies, ratings, on="movieId", how="inner"))
mostRated = moviesRated["title"].value_counts()
mostRated = mostRated.reset_index()
mostRated = mostRated.rename(columns={"title":"count"})
mostRated = mostRated.rename(columns={"index":"title"})

'''
cut off movies with not enough ratings, then cut off movies with avg Rating < 3.5
'''
s = np.array(mostRated["count"])
perc90 = np.percentile(s, 90)
mostRated = mostRated[mostRated["count"] > perc90]
mostRated["avgRating"] = np.nan

tempDf = pd.merge(moviesRated, mostRated, on="title", how="inner").drop("count", axis=1)
tempName = pd.DataFrame()

def get_movie_ratings(titleSearched):
    tempMovieName = tempDf[tempDf["title"] == titleSearched]
    tempMean = round(tempMovieName["rating"].mean(),1)
    #print(titleSearched, " ", tempMean)
    return tempMean

mostRated["avgRating"] = mostRated["title"].apply(lambda x: get_movie_ratings(x) if get_movie_ratings(x) >= 3.5 else np.nan)
mostRated = mostRated.dropna()
print(mostRated)

'''
Tops aquired, use IMDB weighting to get recomms
WeightedRating(WR)= ( (v/(v+m)) ⋅R) + ( (m/(v+m)) ⋅C)
v = number of votes of specific movie : moviesCutSorted"count
m = min vote count Threshold : 0,90 quantile
R = average rating of specific movie : moviesCutSorted"avgRating"
C = mean vote
'''

C = ratings["rating"].mean()
m = perc90

mostRated.head()

def weighted_rating(mostRated):
    v = mostRated["count"]
    R = mostRated["avgRating"]
    return (v/(v+m) * R) + (m/(m+v) * C)

mostRated["weightedScore"] = mostRated.apply(weighted_rating, axis=1)
mostRated = mostRated.sort_values('weightedScore', ascending=False)
mostRated = mostRated.reset_index()
mostRated = mostRated.drop("index", axis=1)
print(mostRated.head(20))

#mostRated.head(10).sort_values("weightedScore").plot.barh(x="title", y="weightedScore")

'''
DONE
EVALUATION - EMPIRICAL
'''
top_movies = pd.read_excel("originalData/TopMovies.xls").copy()
movies_metadata = pd.read_csv("originalData/movies_metadata.csv", usecols=["id", "imdb_id"]).copy()

movies_metadata["id"] = movies_metadata["id"].drop(19730)
movies_metadata["id"] = movies_metadata["id"].drop(29503)
movies_metadata["id"] = movies_metadata["id"].drop(35587)
movies_metadata.isnull().sum()
movies_metadata = movies_metadata.dropna()
movies_metadata["id"] = movies_metadata["id"].astype("int")

movies_metadata.head()
top_movies.head()
mostRated.head()

'''
Determine good recommendations if my top 20 is also inside top_movies
'''
movies = pd.read_csv("exportedData/moviesExp.csv").copy()
my_top = mostRated[:20].merge(movies, on="title", how="inner")
my_top = my_top.merge(movies_metadata, left_on="tmdbId", right_on="id", how="inner")
my_top = pd.DataFrame({
    "title" : my_top["title"],
    "rating" : my_top["weightedScore"],
    "tmdbId" : my_top["id"],
    "imdb_id" : my_top["imdb_id"]
})
my_top
top_movies
len(my_top)
len(top_movies)

positive_count = 0
# my_top vs top_movies
for i in np.arange(len(my_top)):
    for j in np.arange(len(top_movies)):
        if my_top["imdb_id"][i] == top_movies["imdbid"][j]:
            print(my_top["title"][i])
            positive_count += 1

accuracy_score = positive_count / len(my_top)
print(accuracy_score)

