#main libraries
import numpy as np
import pandas as pd
pd.options.display.max_columns = None
pd.options.display.width=None

movies = pd.read_csv("exportedData/moviesExp.csv").copy()
ratings = pd.read_csv("exportedData/ratingsExp.csv").copy()
tags = pd.read_csv("exportedData/tagsExp.csv").copy()

movies = pd.DataFrame({
    "movieId" : movies["movieId"],
    "title" : movies["title"]
})
ratings = pd.DataFrame({
    "movieId" : ratings["movieId"],
    "rating" : ratings["rating"]
})

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
print(mostRated.head(30))

#mostRated.head(10).sort_values("weightedScore").plot.barh(x="title", y="weightedScore")

'''
DONE
'''