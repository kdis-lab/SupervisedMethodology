from skfeature.function.statistical_based import chi_square, f_score, gini_index, t_score
from skfeature.function.similarity_based import reliefF, fisher_score
import multiprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import PowerTransformer
from sklearn.model_selection import StratifiedKFold
import os
import pandas as pd
import numpy as np
np.random.seed(7)

scores = [
    (lambda X, Y: reliefF.feature_ranking(
        reliefF.reliefF(preprocess_data(X), Y)), 'relief'),
    (lambda X, Y: fisher_score.feature_ranking(
        fisher_score.fisher_score(preprocess_data(X), Y)), 'fisher_score'),
    # chi_square needs not-negative data
    (lambda X, Y: chi_square.feature_ranking(
        chi_square.chi_square(preprocess_data(X, max_min=True), Y)), 'chi_square'),
    (lambda X, Y: f_score.feature_ranking(
        f_score.f_score(preprocess_data(X), Y)), 'f_score'),
    (lambda X, Y: t_score.feature_ranking(
        t_score.t_score(preprocess_data(X), Y)), 't_score'),
]

def borda_ranking(rankings):

    n_features = len(rankings[0])
    general = {i: 0 for i in range(n_features)}

    for rank in rankings:

        for index in range(len(rank)):

            key = rank[index]
            # Borda ranking method
            general[key] = general[key] + (n_features - index)

    # sort and reverse, [0] is the best score
    ranking = sorted(general.items(), key=lambda kv: kv[1], reverse=True)

    # normalize / features * len(scores)
    normalize = n_features * len(rankings)
    ranking = np.asarray(
        list(map(lambda kv: np.asarray([kv[0], kv[1]/normalize]), ranking)))

    return ranking

def train_cv(X, Y):

    y = np.copy(Y)
    y[y == 'N'] = 0
    y[y == 'T'] = 1
    y = y.astype(int)
    counts = np.bincount(y)
    paired = counts[0] == counts[1]

    if paired:
        half = X.shape[0]//2
        test = [[i, i+half] for i in range(half)]
        X_in = set(range(X.shape[0]))
        train = [list(X_in - set(ids)) for ids in test]
    else:
        ind = np.argmin(counts)
        folds = counts[ind]
        kfold = StratifiedKFold(n_splits=folds)
        train = [t for t, test in kfold.split(X, y)]

    return train


def preprocess_data(x, max_min=False):

    standardize = True

    if max_min:
        standardize = False

    transformer = PowerTransformer(
        method='yeo-johnson', standardize=standardize)

    X_transf = transformer.fit_transform(x)

    if max_min:

        X_transf = MinMaxScaler().fit_transform(X_transf)

    return X_transf

def generate_ranking(path, scores):

    df = pd.read_csv(path, index_col=0)

    data = df.to_numpy()

    # ignore class
    X = data[:, :-1]
    X = X.astype(float)
    Y = data[:, -1]

    # store rankings peer function peer folds
    results = {s[1]: [] for s in scores}
    results['all'] = []
    indices = train_cv(X, Y)

    for t in indices:

        general_score = []

        for score in scores:

            general_score.append(score[0](X[t], Y[t]))
            results[score[1]].append(general_score[-1])

        # merge scores rankings with Borda method in a fold
        # just keep with the indices, without the Borda scores
        results['all'].append(borda_ranking(general_score)[:, 0])

    # Borda ranking across folds
    for method in results:

        ranking = borda_ranking(results[method])
        
		# add the name of the column depending of the index kv[0]
        ranking = list(map(lambda kv: np.asarray(
            [kv[0], df.columns[kv[0]], kv[1]]), ranking))

        ranking_df = pd.DataFrame(
            columns=['index', 'index_name', 'score'], data=ranking)

        # set index name to avoid the default index 0, ... , N
        ranking_df.set_index('index', inplace=True)

        name = '{}-ranking-{}.csv'.format(path[:path.rfind('.')], method)
        ranking_df.to_csv(name)
        print('end', name)

root = './datasets/Imbalanced'
files = [os.path.join(root, i) for i in os.listdir(root)
         if i.endswith('-preproc-filter.csv')]

# sequential
# for path in files:
# 	generate_ranking(path, scores)

# parallel way

def per_file(path):
    generate_ranking(path, scores)

if __name__ == '__main__':

    with multiprocessing.Pool() as pool:
        results = pool.map(per_file, [path for path in files])