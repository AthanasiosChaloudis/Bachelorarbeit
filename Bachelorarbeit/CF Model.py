#main libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
pd.options.display.max_columns = None
pd.options.display.width=None

#surprise
import surprise
from surprise import Dataset, Reader
from surprise import SVD, KNNBasic, NMF, reader
from surprise.model_selection import cross_validate, KFold, train_test_split


movies = pd.read_csv("exportedData/moviesExp.csv", usecols=["title","movieId"])
ratings = pd.read_csv("exportedData/ratingsExp.csv")
evalData = pd.merge(movies, ratings, on="movieId", how="inner")
df = pd.read_table("exportedData/ratingsExp.csv", sep= ',').drop("timestamp", axis=1)
df.head()
recoms = pd.DataFrame()
reader = surprise.Reader(rating_scale=(0.5,5.0))

df.head()
df_ratings = pd.DataFrame()
df_ratings["ratings_count"] = df.groupby("movieId")["rating"].count().sort_values(ascending=False)
df_ratings = df_ratings.reset_index()
s = np.array(df_ratings["ratings_count"])
perc90 = np.percentile(s, 90)
df_ratings = df_ratings[df_ratings["ratings_count"]>perc90]
check_df = df_ratings.copy()
df_ratings = pd.merge(df_ratings, df, on="movieId", how="inner")

print("We work with : " +str(len(df_ratings))+ " movies, as the rest " + str(len(df)-len(df_ratings))+ " received less than "+ str(perc90)+ " ratings.")

df_ratings = df_ratings.drop("ratings_count", axis=1)

data = surprise.Dataset.load_from_df(df_ratings, reader)

trainset = data.build_full_trainset()
#Using KNNBasic algorithm
alg = surprise.SVD()
#Training model
alg.fit(trainset)


def recom(uid, recomms_count):
    movieIds = df["movieId"].unique()
    rated_movies = df.loc[df["userId"] == uid, "movieId"]
    iid_to_pred = np.setdiff1d(movieIds, rated_movies)

    test_data = [[uid, iid, 5.0] for iid in iid_to_pred]
    predictions = alg.test(test_data)

    pred_ratings = np.array([pred.est for pred in predictions])
    indice_max = np.argpartition(pred_ratings, -recomms_count)[-recomms_count:]
    iid = iid_to_pred[indice_max]

    iid_to_title = [i for i in range(0, recomms_count)]
    iid_ratings = [i for i in range(0, recomms_count)]

    for i in range(len(iid)):
        temp = iid[i]
        temp = movies[movies["movieId"] == temp]
        temp = temp.reset_index()
        temp = temp["title"][0]
        iid_to_title[i] = temp
        iid_ratings[i] = pred_ratings[indice_max][i]

    return iid_to_title, iid_ratings

moviesExtras = pd.read_csv("exportedData/moviesExp.csv", usecols=["title","Year", "genres_list"])

def recomDf(uid, recoms=recoms):
    recoms["title"] = recom(uid, 10)[0]
    recoms["predicted_score"] = recom(uid, 10)[1]
    recoms = recoms.merge(moviesExtras, on="title", how="inner")
    recoms = recoms.sort_values(by="predicted_score", ascending=False)
    print(recoms)
    return recoms

recom(52, 10)
recomDf(52)

#############
# ADD YEAR AND ONE GENRE
#############
moviesTemp = pd.read_csv("exportedData/moviesExp.csv", usecols=["title", "Action", "movieId"]).merge(ratings.groupby("movieId").rating.count(), on="movieId", how="inner").\
    merge(ratings.groupby("movieId").rating.mean(), on="movieId", how="inner")
actionMovies = moviesTemp[moviesTemp["Action"] == True]
actionMovies
actionMovies = actionMovies[actionMovies["rating_x"] > 30]
actionMovies = actionMovies[actionMovies["rating_y"]> 3.5]
len(actionMovies)

#temp = movies[movies["title"] == "Princess Bride, The (1987)"]
temp = movies[movies["title"] == "Dark Knight, The (2008)"]
temp["movieId"]
check_df[check_df["movieId"] == 1197]
ratings[ratings["userId"] == 52]
