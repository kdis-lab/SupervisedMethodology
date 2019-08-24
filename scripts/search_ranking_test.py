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


def files_rankings(root, name):

    arr = [os.path.join(root, i) for i in os.listdir(
        root) if i.startswith(name[:name.rfind('.')] + '-')]
    arr.sort()
    return arr


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


def one_fold(model, X_train, y_train, X_test, y_test):

    print('start fold')

    sc = make_scorer(roc_auc_score)

    transformer = PowerTransformer(
        method='yeo-johnson', standardize=True)

    transformer.fit(X_train)

    clf = GridSearchCV(model["model"], model["parameters"],
                       cv=3, n_jobs=-1, scoring=sc)

    clf.fit(transformer.transform(X_train), y_train)

    y_predic = clf.predict(transformer.transform(X_test))

    print('end fold')

    return roc_auc_score(y_test, y_predic)


def eval_model(model, X, y, pFolds=None):

    folds = train_cv(X, y) if pFolds is None else pFolds

    # scores = [one_fold(model, X, y, train, test) for train, test in folds]
    with multiprocessing.Pool() as pool:
        scores = pool.starmap(
            one_fold, [(model, X[train], y[train], X[test], y[test]) for train, test in folds])

    results = np.asarray(scores)

    return np.round(results.mean(), decimals=3), np.round(results, decimals=3), folds


def process_file(pathD, pathRs):

    print('init file', pathD)
    df = pd.read_csv(pathD, index_col=0)

    # Transforming the class att
    df['Class'] = df['Class'].apply(
        {'N': 0, 'T': 1}.get)

    for path_rank in pathRs:

        print(path_rank)
        dfR = pd.read_csv(path_rank, index_col=1)
        indices = dfR.index.values

        for model in models:

            print('model', model['model_name'])
            
            results = pd.DataFrame(
                columns=['ID', 'NumberAtts', 'Atts', 'metricOpt'])

            results.set_index('ID', inplace=True)

            filedir = os.path.join(root, 'results', pathD[pathD.rfind('/')+1:].replace('.csv', ''),
                                   path_rank[path_rank.rfind('-')+1:].replace('.csv', ''))

            reportDir = os.path.join(filedir, model["model_name"] + '.csv')

            if os.path.exists(reportDir):
                continue

            columns = [indices[0]]

            general_value, general_scores, folds = eval_continue(
                df, model, columns, pFolds=None)

            results = results.append({
                'NumberAtts': len(columns),
                'Atts': ' '.join(columns),
                'metricOpt': general_value}, ignore_index=True)

            if general_value == 1:
                break

            for i in range(1, attr_limit):

                print('attr', i)
                new_columns = columns+[indices[i]]

                curr_value, curr_scores, c_folds = eval_continue(
                    df, model, new_columns, pFolds=folds)

                if (general_value < curr_value) and wilcoxon(general_scores, curr_scores, alternative='less').pvalue <= 0.05:

                    general_value = curr_value
                    columns = new_columns

                    results = results.append({
                        'NumberAtts': len(columns),
                        'Atts': ' '.join(columns),
                        'metricOpt': general_value}, ignore_index=True)

                if general_value == 1:
                    break

            if not os.path.exists(filedir):
                os.makedirs(filedir)

            # Saving the results
            results.sort_values(['metricOpt', 'NumberAtts'], ascending=[
                                False, True], inplace=True)
            results.to_csv(reportDir)


attr_limit = 100

models = [{"model_name": "SVM",
           "model": SVC(gamma="auto"),
           "parameters":  {
               # types of kernels to be tested
               "kernel": ["linear", "poly", "rbf", "sigmoid"],
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

root = './3'
files = [(os.path.join(root, i), files_rankings(root, i))
         for i in os.listdir(root) if i.endswith('-filter.csv')]

for pathD, pathRs in files:
    process_file(pathD, pathRs)
