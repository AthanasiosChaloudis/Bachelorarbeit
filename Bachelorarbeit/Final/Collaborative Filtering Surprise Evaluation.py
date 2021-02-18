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
from surprise import SVD, KNNBasic, reader
from surprise.model_selection import cross_validate, KFold, train_test_split

#######################################
# NOT EDITED AT ALL, JUST COPY PASTED #
#######################################


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
user_based = {"name": "cosine",
                      "user_based" : True}
item_based = {"name": "cosine",
                      "user_based" : False}

trainset = data.build_full_trainset()
#Using KNNBasic algorithm
alg_i = surprise.KNNBasic(sim_options=item_based)
alg_u = surprise.KNNBasic(sim_options=user_based)

#cv = cross_validate(alg_i, data, measures=['RMSE', 'MAE', "MSE"], cv=5, verbose=True)


def plot_Errors(cv):
    rmse = cv["test_rmse"]
    mae = cv["test_mae"]
    mse = cv["test_mse"]
    x = np.arange(len(rmse))

    # PLOT!
    fig, ax = plt.subplots(figsize=(10, 5))
    plt.xticks(np.arange(min(x), max(x) + 1, 1.0))
    plt.ylim(0.5, 1.0)
    ax.plot(x, rmse, marker='o', label="RMSE")
    ax.plot(x, mae, marker='o', label="MAE")
    ax.plot(x, mse, marker='o', label="MSE")


    # Chart setup
    plt.title("Model Evaluative Metrics", fontsize=12)
    plt.xlabel("Cross Validation Folds", fontsize=10)
    plt.ylabel("Metrics Scores", fontsize=10)
    plt.legend()
    plt.show()

def plot_times(cv):
    fit_time = cv["fit_time"]
    test_time = cv["test_time"]
    x = np.arange(len(fit_time))

    fig, ax = plt.subplots(figsize=(10, 5))
    plt.xticks(np.arange(min(x), max(x) + 1, 1.0))
    ax.plot(x, fit_time, marker='o', label="Fit Time")
    ax.plot(x, test_time, marker='o', label="Test Time")

    # Chart setup
    plt.title("Model Time Evaluation", fontsize=12)
    plt.xlabel("Cross Validation Folds", fontsize=10)
    plt.ylabel("Times in seconds", fontsize=10)
    plt.legend()
    plt.show()

benchmark = []
# Iterate over all algorithms --> First Fold ist train, k-1 Folds for testing
#for algorithm in [SVD(), KNNBasic()]:
algorithm = KNNBasic()
# Perform cross validation
results = cross_validate(algorithm, data, measures=['RMSE', 'MAE', "MSE"], cv=3, verbose=False)

# Get results & append algorithm name
tmp = pd.DataFrame.from_dict(results).mean(axis=0)
tmp = tmp.append(pd.Series([str(algorithm).split(' ')[0].split('.')[-1]], index=['Algorithm']))
benchmark.append(tmp)
plot_Errors(results)
#plot_times(results)

pd.DataFrame(benchmark).set_index('Algorithm').sort_values('test_rmse')

benchmark[0]

kf = KFold(n_splits=3)

from collections import defaultdict

def precision_recall_at_k(predictions, k=3, threshold=3.5):
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

def prec_rec_threshold(alg):
    final = []
    trainset = data.build_full_trainset()
    alg.fit(trainset)

    testset = trainset.build_testset()
    predictions = alg.test(testset)
    for threshold in np.arange(0, 5.5, 0.5):
        tp = 0
        fn = 0
        fp = 0
        tn = 0
        temp = []
        for uid, _, true_r, est, _ in predictions:
            if (true_r >= threshold):
                if (est >= threshold):
                    tp = tp + 1
                else:
                    fn = fn + 1
            else:
                if (est >= threshold):
                    fp = fp + 1
                else:
                    tn = tn + 1
            if tp == 0:
                precision = 0
                recall = 0
                f1 = 0
            else:
                precision = tp / (tp + fp)
                recall = tp / (tp + fn)
                f1 = 2 * (precision * recall) / (precision + recall)
        temp = [threshold, tp, fp, tn, fn, precision, recall, f1]
        final.append(temp)
    results = pd.DataFrame(final)
    results.rename(columns={0: 'threshold', 1: 'tp', 2: 'fp', 3: 'tn', 4: 'fn', 5: 'Precision', 6: 'Recall', 7: 'F1'},
                   inplace=True)
    return results

#prec_rec_threshold(SVD())

def prec_rec_test(alg):
    precList = []
    recList = []
    f1scores = []
    for trainset, testset in kf.split(data):
        alg.fit(trainset)
        predictions = alg.test(testset)
        precisions, recalls = precision_recall_at_k(predictions, k=3, threshold=3.5)
        precList.append(sum(prec for prec in precisions.values()) / len(precisions))
        recList.append(sum(rec for rec in recalls.values()) / len(recalls))

    for i in range(len(precList)):
        f1scores.append(2 * ((precList[i] * recList[i]) / (precList[i] + recList[i])))

    f1_mean = sum(f1scores)/len(f1scores)
    precision_mean = sum(precList) / len(precList)
    recall_mean = sum(recList) / len(recList)
    result = precision_mean, recall_mean, f1_mean

    plot_PRF(precList, recList, f1scores)

    return result

def plot_PRF(precList, recList, f1scores):
    x = np.arange(len(precList))
    # PLOT!
    fig, ax = plt.subplots(figsize=(10, 5))
    plt.xticks(np.arange(min(x), max(x) + 1, 1.0))
    plt.yticks(np.arange(min(x), max(x), step=0.1))
    plt.ylim(0, 1)
    ax.plot(x, precList, marker='o', label="Precision")
    ax.plot(x, recList, marker='o', label="Recall")
    ax.plot(x, f1scores, marker='o', label="F1-Scores")

    # Chart setup
    plt.title("Precision, Recall, F1-Scores", fontsize=12)
    plt.xlabel("Cross Validation Folds", fontsize=10)
    plt.ylabel("Score", fontsize=10)
    plt.legend()
    plt.show()

prec_rec_test(KNNBasic())

#plot_PRF(SVD())

def check_k_and_thresh(algo):
    global predictions
    prec_to_ave = []
    rec_to_ave = []
    kf = KFold(n_splits=30)

    for trainset, testset in kf.split(data):
        algo.fit(trainset)
        predictions = algo.test(testset)
        precisions, recalls = precision_recall_at_k(predictions, k=30, threshold=2.5)

        # Precision and recall can then be averaged over all users

        prec_to_ave.append(sum(prec for prec in precisions.values()) / len(precisions))
        rec_to_ave.append(sum(rec for rec in recalls.values()) / len(recalls))

    results = []
    for i in range(2, 30):
        precisions, recalls = precision_recall_at_k(predictions, k=i, threshold=2.5)

        # Precision and recall can then be averaged over all users
        prec = sum(prec for prec in precisions.values()) / len(precisions)
        rec = sum(rec for rec in recalls.values()) / len(recalls)
        results.append({'K': i, 'Precision': prec, 'Recall': rec})

    K = np.arange(2,30)
    precs = []
    recs = []
    for i in range(len(K)):
        precs.append(results[i].get("Precision"))
        recs.append(results[i].get("Recall"))

    plt.plot(K, precs)
    plt.plot(K, recs)

def run():
    for algorithm in [SVD(), KNNBasic()]:
        prec_rec_test(algorithm)
    for algorithm in [SVD(), KNNBasic()]:
        plot_PRF(algorithm)


