#main libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
pd.options.display.max_columns = None
pd.options.display.width=None
from collections import defaultdict
from surprise import Dataset

#sklearn
from sklearn.model_selection import train_test_split, KFold
from scipy import sparse
import sklearn.metrics.pairwise as pw

#surprise
import surprise
from surprise import SVD, KNNBasic, NMF, reader
from surprise.model_selection import cross_validate, KFold

'''
movies = pd.read_csv("exportedData/moviesExp.csv", usecols=["title","movieId"])
ratings = pd.read_csv("exportedData/ratingsExp.csv", usecols=["movieId","rating", "userId"])
df = pd.merge(movies, ratings, on="movieId", how="inner")
rating_matrix = df.pivot_table(index="userId", columns="movieId", values="rating")
sparse_matrix = sparse.csr_matrix(rating_matrix)
'''
movies = pd.read_csv("exportedData/moviesExp.csv", usecols=["title","movieId"])
ratings = pd.read_csv("exportedData/ratingsExp.csv")
evalData = pd.merge(movies, ratings, on="movieId", how="inner")
df = pd.read_table("exportedData/ratingsExp.csv", sep= ',').drop("timestamp", axis=1)
df.head()
recoms = pd.DataFrame()
reader = surprise.Reader(rating_scale=(0.5,5.0))
data = surprise.Dataset.load_from_df(df, reader)

trainset, testset = train_test_split(data, test_size=0.25)

#alg = surprise.SVD()
alg = surprise.KNNBasic()
out = alg.fit(data.build_full_trainset())

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

def recomDf(uid, recoms=recoms):
    recoms["title"] = recom(uid, 5)[0]
    recoms["predicted_score"] = recom(uid, 5)[1]
    recoms = recoms.sort_values(by="predicted_score", ascending=False)
    print(recoms)
    return recoms

recom(52, 5)
recomDf(52)

alg1 = surprise.SVD()
alg2 = surprise.KNNBasic()
alg3 = surprise.NMF()

#cross_validate(alg1, data, measures=['RMSE', 'MAE', "MSE"], cv=5, verbose=True)
#cross_validate(alg2, data, measures=['RMSE', 'MAE', "MSE"], cv=5, verbose=True)
#cross_validate(alg3, data, measures=['RMSE', 'MAE', "MSE"], cv=5, verbose=True)


##############
# EVALUATION #
##############

benchmark = []
# Iterate over all algorithms --> First Fold ist train, k-1 Folds for testing
for algorithm in [SVD(), NMF(), KNNBasic()]:
    # Perform cross validation
    results = cross_validate(algorithm, data, measures=['RMSE', 'MAE', "MSE"], cv=5, verbose=False)

    # Get results & append algorithm name
    tmp = pd.DataFrame.from_dict(results).mean(axis=0)
    tmp = tmp.append(pd.Series([str(algorithm).split(' ')[0].split('.')[-1]], index=['Algorithm']))
    benchmark.append(tmp)

pd.DataFrame(benchmark).set_index('Algorithm').sort_values('test_rmse')

def precision_recall_at_k(predictions, k=10, threshold=3.5):
    """Return precision and recall at k metrics for each user"""

    # First map the predictions to each user.
    user_est_true = defaultdict(list)
    for uid, _, true_r, est, _ in predictions:
        user_est_true[uid].append((est, true_r))

    precisions = dict()
    recalls = dict()
    for uid, user_ratings in user_est_true.items():

        # Sort user ratings by estimated value
        user_ratings.sort(key=lambda x: x[0], reverse=True)

        # Number of relevant items
        n_rel = sum((true_r >= threshold) for (_, true_r) in user_ratings)

        # Number of recommended items in top k
        n_rec_k = sum((est >= threshold) for (est, _) in user_ratings[:k])

        # Number of relevant and recommended items in top k
        n_rel_and_rec_k = sum(((true_r >= threshold) and (est >= threshold))
                              for (est, true_r) in user_ratings[:k])

        # Precision@K: Proportion of recommended items that are relevant
        # When n_rec_k is 0, Precision is undefined. We here set it to 0.

        precisions[uid] = n_rel_and_rec_k / n_rec_k if n_rec_k != 0 else 0

        # Recall@K: Proportion of relevant items that are recommended
        # When n_rel is 0, Recall is undefined. We here set it to 0.

        recalls[uid] = n_rel_and_rec_k / n_rel if n_rel != 0 else 0

    return precisions, recalls


#data = Dataset.load_builtin('ml-100k')
data = ratings.drop("timestamp", axis=1)
data = surprise.Dataset.load_from_df(data, reader)
kf = KFold(n_splits=5)
algo = SVD()

precision_mean = 0
recall_mean = 0

for trainset, testset in kf.split(data):
    alg.fit(trainset)
    predictions = alg.test(testset)
    precisions, recalls = precision_recall_at_k(predictions, k=5, threshold=3.5)

    # Precision and recall can then be averaged over all users
    print("New Fold \n")
    print("Precision : "+ str(sum(prec for prec in precisions.values()) / len(precisions)) )
    print("Recall : "+ str(sum(rec for rec in recalls.values()) / len(recalls)) )

    precision_mean = precision_mean + sum(prec for prec in precisions.values()) / len(precisions)
    recall_mean = recall_mean + sum(rec for rec in recalls.values()) / len(recalls)

precision_mean = precision_mean/5
recall_mean = recall_mean/5


#######
# Other evaluation: hide 1 or 3 movies from uid and check then if the recom recommends them at the end

# find users with most ratings and use them to find TP TN FP FN!!!!
# print also F1 :D
#######
