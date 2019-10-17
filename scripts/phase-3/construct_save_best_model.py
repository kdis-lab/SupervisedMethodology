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
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import make_scorer
from CustomGrid import CustomGrid
import pickle

np.random.seed(7)

def eval_continue(df, model, columns, pFolds=None):

    # ignore class
    X = df.loc[:, columns].to_numpy()
    Y = df.loc[:, ['Class']].to_numpy()[:, 0]

    avg_folds, scores, folds = eval_model(model, X, Y, pFolds)

    return avg_folds, scores, folds


def train_cv(X, Y):

    counts = np.bincount(Y)
    paired = counts[0] == counts[1]

    if paired:

        half = X.shape[0]//2
        test_sets = [[i, i+half] for i in range(half)]
        all_instances = set(range(X.shape[0]))
        indexes = [(list(all_instances - set(test)), test)
                   for test in test_sets]
    else:

        ind = np.argmin(counts)
        folds = counts[ind]
        kfold = StratifiedKFold(n_splits=folds)
        indexes = [(train, test) for train, test in kfold.split(X, Y)]

    return indexes


def one_fold(model, X_train, y_train, X_test, y_test, return_model= False):

    print('start fold')

    transformer = PowerTransformer(
        method='yeo-johnson', standardize=True)

    transformer.fit(X_train)

    clf = CustomGrid(model["model"], model["parameters"], 3)

    clf.fit(transformer.transform(X_train), y_train)

    y_predic = clf.predict(transformer.transform(X_test))

    print('end fold')

    if not return_model:
        return roc_auc_score(y_test, y_predic)
    else:
        return clf, roc_auc_score(y_test, y_predic)


def eval_model(model, X, y, pFolds=None):

    folds = train_cv(X, y) if pFolds is None else pFolds
 
    scores = [one_fold(model, X[train], y[train], X[test], y[test]) for train, test in folds]

    results = np.asarray(scores)

    return np.round(results.mean(), decimals=3), np.round(results, decimals=3), folds


def process_file(pathD, resultsDir):

    name_dataset = pathD.split("/")[1]

    df = pd.read_csv("datasets/" + name_dataset + ".csv", index_col=0)

    df['Class'] = df['Class'].apply(
        {'N': 0, 'T': 1}.get)

    # For each file
    for model in models:

        print('model', model['model_name'])

        best_subsets = pd.read_csv(pathD + model['model_name'] + ".csv")

        subset= best_subsets.iloc[0, 2]

        features= subset.split(' ')

        filedir = os.path.join(resultsDir, pathD[pathD.find('/')+1:pathD.rfind('/')+1], model['model_name'])
        
        if not os.path.exists(filedir):
            os.makedirs(filedir)

        general_value, general_scores, folds = eval_continue(
                df, model, features, pFolds=None)

        index_best = np.argmax(general_scores)

        train_indexes, test_indexes= folds[index_best]

        X = df.loc[:, features].to_numpy()
        Y = df.loc[:, ['Class']].to_numpy()[:, 0]

        clf, roc_auc_test = one_fold(model, X[train_indexes], Y[train_indexes], X[test_indexes], Y[test_indexes], return_model= True)

        assert general_scores[index_best] == roc_auc_test, "Different results"

        pickle.dump(clf, open(filedir + "/" + model['model_name'] + ".model", "wb"))

        pickle.dump(test_indexes, open(filedir + "/" + model['model_name'] + ".test_indexes", "wb"))

        pickle.dump(train_indexes, open(filedir + "/" + model['model_name'] + ".train_indexes", "wb"))

models = [
           {"model_name": "SVM",
           "model": SVC(gamma="auto", probability= True),
           "parameters":  {
               # types of kernels to be tested
               "kernel": ["rbf", "sigmoid"],
               "C": [0.01, 0.1, 1, 10],  # range of C to be tested
               "degree": [1, 2, 3]  # degrees to be tested
           }},
           {"model_name": "RF",
           "model": RandomForestClassifier(),
           "parameters":  {
               'max_features': ['auto',
                                'sqrt'],  # Number of features to consider at every split
               'min_samples_split':
               [2, 5, 10],  # Minimum number of samples required to split a node
               'min_samples_leaf':
               [1, 2, 4],  # Minimum number of samples required at each leaf node
               'criterion': ["gini", "entropy"]  # criteria to be tested
           }
           },

          {"model_name": "LR",
           "model": LogisticRegression(),
           "parameters":  {
               # regularization hyperparameter space
               'C': np.logspace(0, 4, 5),
               'penalty': ['l1', 'l2']  # regularization penalty space
           }}
          ]

path = "results/SKCM-preproc-filter/all/"

process_file(path, "binary_models/")





