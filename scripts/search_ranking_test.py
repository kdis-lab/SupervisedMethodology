import multiprocessing
from scipy.stats import wilcoxon
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.base import clone
import sklearn
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import PowerTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
np.random.seed(7)
import datetime


def files_rankings(root, name):
    arr = [os.path.join(root, i) for i in os.listdir(
        root) if i.startswith(name[:name.rfind('.')] + '-')]
    arr.sort()
    return arr


def new_dataframe(df, columns):
    cols = columns + ['Class']
    return df[cols]


def to_numpy(df, columns):
    new_df = new_dataframe(df, columns)
    data = new_df.to_numpy()
    # ignore class
    X = data[:, :-1]
    X = X.astype(float)
    Y = data[:, -1]
    return X, Y


def save_results(results, columns, metric): return results.append({
    'NumberAtts': len(columns),
    'Atts': ' '.join(columns),
    'metricOpt': metric}, ignore_index=True)


def eval_continue(df, model, columns, pFolds=None):
    X, y = to_numpy(df, columns)
    avg_folds, scores, folds = eval_model(model, X, y, pFolds)
    return avg_folds, scores, folds


def to_binary(x):
    x[x == 'N'] = 0
    x[x == 'T'] = 1
    return x.astype(int)


def train_cv(X, Y):

    y = np.copy(Y)
    y = to_binary(y)
    counts = np.bincount(y)
    paired = counts[0] == counts[1]

    if paired:
        half = X.shape[0]//2
        test = [[i, i+half] for i in range(half)]
        X_in = [i for i in range(X.shape[0])]
        train = [(list(set(X_in) - set(ids)), ids) for ids in test]
    else:
        ind = np.argmin(counts)
        folds = counts[ind]
        kfold = StratifiedKFold(n_splits=folds)
        train = [(t, test) for t, test in kfold.split(X, y)]

    return train


def one_fold(model, X, y, train, test):
    print('start fold')
    sc = make_scorer(roc_auc_score)

    transformer = PowerTransformer(
        method='yeo-johnson', standardize=True)

    transformer.fit(X[train])

    clf = GridSearchCV(model["model"], model["parameters"],
                       cv=5, n_jobs=-1, scoring=sc)

    clf.fit(transformer.transform(X[train]), to_binary(y[train]))

    y_predic = clf.predict(transformer.transform(X[test]))
    y_true = to_binary(y[test])
    print('end fold')

    return roc_auc_score(y_true, y_predic)


def eval_model(model, X, y, pFolds=None):

    folds = train_cv(X, y) if pFolds is None else pFolds

    scores = [one_fold(model, X, y, train, test) for train, test in folds]
    #with multiprocessing.Pool() as pool:
    #    scores = pool.starmap(
    #        one_fold, [(model, X, y, train, test) for train, test in folds])

    return np.asarray(scores).mean(), np.asarray(scores), folds


def process_file(pathD, pathRs):

    print('init file', pathD)
    df = pd.read_csv(pathD, index_col=0)

    for path_rank in pathRs:

        print(path_rank)
        dfR = pd.read_csv(path_rank, index_col=1)
        indices = dfR.index.values

        for model in models:

            print('model', model['model_name'])
            results = pd.DataFrame(
                columns=['ID', 'NumberAtts', 'Atts', 'metricOpt'])
            results.set_index('ID', inplace=True)

            # se puso aqui en 1 para limitar el tiempo de busqueda
            for i in range(1):
                columns = [indices[i]]

                general_value, general_scores, folds = eval_continue(
                    df, model, columns, pFolds=None)
                results = save_results(results, columns, general_value)

                if general_value == 1:
                    break

                # se redujo por tiempo
                for j in range(i+1, attr_limit):

                    print('attr', j)
                    new_columns = columns+[indices[j]]

                    curr_value, curr_scores, c_folds = eval_continue(
                        df, model, new_columns, pFolds=folds)

                    if np.array_equal(general_scores, curr_scores):
                        continue

                    if (general_value < curr_value):
                    	if (curr_value == 1) or (wilcoxon(general_scores, curr_scores, alternative='less').pvalue <= 0.05):
                            general_value = curr_value
                            columns = new_columns
                            results = save_results(results, columns, general_value)
                            if general_value == 1:
                                break
                if general_value == 1:
                    break

            filedir = os.path.join(root, 'results', pathD[pathD.rfind('/')+1:].replace('.csv', ''),
                                   path_rank[path_rank.rfind('-')+1:].replace('.csv', ''))

            if not os.path.exists(filedir):
                os.makedirs(filedir)

            results.index.name = 'ID'
            results.sort_values(['metricOpt', 'NumberAtts'], ascending=[
                                False, True], inplace=True)
            results.to_csv(os.path.join(filedir, model["model_name"] + '.csv'))
            


attr_limit = 50

models = [{"model_name": "SVM",
           "model": SVC(gamma="auto"),
           "parameters":  {
               # types of kernels to be tested
               "kernel": ["linear", "poly", "rbf", "sigmoid"],
               "C": [0.01, 0.1, 1, 10],  # range of C to be tested
               "degree": [1, 2, 3]  # degrees to be tested
           }},

          #{"model_name": "RF",
           #"model": RandomForestClassifier(),
           #"parameters":  {
           #    'n_estimators': [int(x) for x in np.linspace(start=10, stop=100, num=10)
           #                     ],  # Number of trees in random forest
           #    'max_features': ['auto',
           #                     'sqrt'],  # Number of features to consider at every split
           #    'max_depth': [int(x) for x in np.linspace(10, 50, num=10)
           #                  ],  # Maximum number of levels in tree
           #    'min_samples_split':
           #    [2, 5, 10],  # Minimum number of samples required to split a node
           #    'min_samples_leaf':
           #    [1, 2, 4],  # Minimum number of samples required at each leaf node
           #    'bootstrap': [True,
           #                  False],  # Method of selecting samples for training each tree
           #    'criterion': ["gini", "entropy"]  # criteria to be tested
           #}
           #},

          {"model_name": "LR",
           "model": LogisticRegression(),
           "parameters":  {
               # regularization hyperparameter space
               'C': np.logspace(0, 4, 5),
               'penalty': ['l1', 'l2']  # regularization penalty space
           }}
          ]

root = './3'
files = [(os.path.join(root, i), files_rankings(root, i))
         for i in os.listdir(root) if i.endswith('-filter.csv')]

for pathD, pathRs in files:
    process_file(pathD, pathRs)
