# coding=utf-8
'''
Consists of 2 different alternatives --> user-based and item-based

In user-based, similar users which have similar ratings for similar items are found
and then target user’s rating for the item which target user has never interacted is predicted!!
1    Specify the target user (which is Bob in this example)
2   Find similar users who have similar ratings to target user (can be more than one)
3    Extract the items which target user never interacted
4    Predict the ratings of unobserved items for target user
5    If the predicted ratings are above the threshold, then recommend them to target user

On the other hand, item-based models find similar items to items which target user already rated or interacted.
1    Specify the target user
2    Find similar items which have similar ratings with items target user rated
3    Predict the ratings for similar items
4    If the predicted ratings are above the threshold, then recommend them to target user
'''
#main libraries
import numpy as np
import pandas as pd
#import seaborn as sns
#import matplotlib.pyplot as plt
pd.options.display.max_columns = None
pd.options.display.width=None

#sklearn
from sklearn.model_selection import train_test_split
from scipy import sparse
import sklearn.metrics.pairwise as pw

movies = pd.read_csv("exportedData/moviesExp.csv", usecols=["title","movieId"])
ratings = pd.read_csv("exportedData/ratingsExp.csv", usecols=["movieId","rating", "userId"])
df = pd.merge(movies, ratings, on="movieId", how="inner")

rating_matrix = df.pivot_table(index="userId", columns="movieId", values="rating")
df_ratings = pd.DataFrame()
df_ratings["ratings_count"] = df.groupby("title")["rating"].count().sort_values(ascending=False)
df_ratings = df_ratings.reset_index()
s = np.array(df_ratings["ratings_count"])
perc90 = np.percentile(s, 90)
perc75 = np.percentile(s, 75)
df_ratings = df_ratings[df_ratings["ratings_count"]>perc90]

df_ratings.head()



len(df_ratings)
df_ratings = pd.merge(df_ratings, df, on="title", how="inner")
df_ratings.head()
df.head()
print("We work with : " +str(len(df_ratings))+ " movies, as the rest " + str(len(df)-len(df_ratings))+ " received less than "+ str(perc90)+ " ratings.")

'''
user / item - based recom
'''
def item_based_recom(input_dataframe,film_name):
    pivot_item_based = pd.pivot_table(input_dataframe, index='title', columns=['userId'], values='rating')
    sparse_pivot = sparse.csr_matrix(pivot_item_based.fillna(0))
    recommender = pw.cosine_similarity(sparse_pivot)
    recommender_df = pd.DataFrame(recommender, columns=pivot_item_based.index, index=pivot_item_based.index)
    ## Item Rating Based Cosine Similarity
    cosine_df = pd.DataFrame(recommender_df[film_name].sort_values(ascending=False))
    cosine_df.reset_index(level=0, inplace=True)
    cosine_df.columns = ['title','cosine_sim']
    return cosine_df.head(5)

result = pd.DataFrame(data=item_based_recom(df, 'Dark Knight, The (2008)'))
result
#result.sort_values("cosine_sim").plot.barh(x="title", y="cosine_sim")

print(item_based_recom(df, 'Dark Knight, The (2008)'))

def user_based_recom(input_dataframe,input_user_id):
    pivot_user_based = pd.pivot_table(input_dataframe, index='title', columns=['userId'], values='rating').T
    sparse_pivot_ub = sparse.csr_matrix(pivot_user_based.fillna(0))
    user_recomm = pw.cosine_similarity(sparse_pivot_ub)
    user_recomm_df = pd.DataFrame(user_recomm, columns=pivot_user_based.index.values,index=pivot_user_based.index.values)
    ## Item Rating Based Cosine Similarity
    usr_cosine_df = pd.DataFrame(user_recomm_df[input_user_id].sort_values(ascending=False))
    usr_cosine_df.reset_index(level=0, inplace=True)
    usr_cosine_df.columns = ['userId','cosine_sim']
    return usr_cosine_df

def recommend_item(user_index, similar_user_indices, matrix, items=5):
    # load vectors for similar users
    similar_users = matrix[matrix.index.isin(similar_user_indices)]  # calc avg ratings across the 3 similar users
    similar_users = similar_users.mean(axis=0)  # convert to dataframe so its easy to sort and filter
    similar_users_df = pd.DataFrame(similar_users, columns=['mean'])

    # load vector for the current user
    user_df = matrix[matrix.index == user_index]  # transpose it so its easier to filter
    user_df_transposed = user_df.transpose()  # rename the column as 'rating'
    user_df_transposed.columns = ['rating']  # remove any rows without a 0 value. Anime not watched yet
    user_df_transposed = user_df_transposed[
        user_df_transposed['rating'] == 0]  # generate a list of animes the user has not seen
    animes_unseen = user_df_transposed.index.tolist()

    # filter avg ratings of similar users for only anime the current user has not seen
    similar_users_df_filtered = similar_users_df[similar_users_df.index.isin(animes_unseen)]  # order the dataframe
    similar_users_df_ordered = similar_users_df.sort_values(by=['mean'], ascending=False)  # grab the top n anime
    top_n_anime = similar_users_df_ordered.head(items)
    top_n_anime_indices = top_n_anime.index.tolist()  # lookup these anime in the other dataframe to find names
    anime_information = movies[movies['movieId'].isin(top_n_anime_indices)]
    anime_information = anime_information.merge(top_n_anime, how="inner", on="movieId")
    anime_information.columns = ["movieId", "title", "Predicted_Rating"]

    return anime_information  # items

user_id = 52

similar_user_indices = user_based_recom(df_ratings, user_id).head(5)["userId"].tolist()
print(similar_user_indices)
#print(recommend_item(user_id, similar_user_indices, rating_matrix))


##############
# EVALUATION #
##############

# Print movies uid actually saw and rated 5 over .4 5 under .2,5

'''
Precision and Recall --> ROC and AUC
elevant items >=3.5 rating

TP = recommended and r > 3.5
TN = not recommended and r < 3.5  


'''
