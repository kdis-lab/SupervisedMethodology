from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn import preprocessing
from sklearn import datasets
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
import pandas as pd
import numpy as np
from sklearn.metrics import matthews_corrcoef, make_scorer
from sklearn.svm import SVC
from scipy.stats import randint as sp_randint
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Lasso

class MyLasso(Lasso):
	def predict(self, X):
		return super().predict(X).round().astype(int)
np.random.seed(7)

from gcready import MyGCForest, get_toy_config

datasets = {
    'BLCA': "datasets/BLCA-genes.txt_genunnormalized-filtered.csv",
    'CHOL': "datasets/CHOL-genes.txt_genunnormalized-filtered.csv",
    'ESCA': "datasets/ESCA-genes.txt_genunnormalized-filtered.csv",
    'LIHC': "datasets/LIHC-genes.txt_genunnormalized-filtered.csv",
    'READ': "datasets/READ-genes.txt_genunnormalized-filtered.csv",
    'SKCM': "datasets/SKCM-genes.txt_genunnormalized-filtered.csv",
    'STAD': "datasets/STAD-genes.txt_genunnormalized-filtered.csv",
#    'STES': "datasets/STES-genes.txt_genunnormalized-filtered.csv",
    'UCEC': "datasets/UCEC-genes.txt_genunnormalized-filtered.csv"
}

classifiers = {
    #"lr": LogisticRegression,
    #"svm": SVC,
    #"kNN": KNeighborsClassifier,
    #"rf": RandomForestClassifier,
    #"lasso": MyLasso,
	"gc": MyGCForest
}

param_grid = {}

matt = make_scorer(matthews_corrcoef)

columns = ['Dataset'] + [name for name, classifier in classifiers.items()]

dataframe_results = pd.DataFrame(columns=columns)
dataframe_results.set_index("Dataset", inplace=True)

for nameDataset, path in datasets.items():

    dataset = pd.read_csv(path)

    X = np.array(dataset[dataset.columns[:-1]], dtype=np.float)

    classes = np.unique(dataset["Class"].values)

    Y = 1 * (dataset["Class"].values == classes[0])

    n_splits = dataset["Class"].value_counts().values.min()

    for name, classifier in classifiers.items():

        results = cross_val_score(
            MyGCForest(get_toy_config()),
            X=X,
            y=Y,
            cv= StratifiedKFold(n_splits= n_splits, shuffle=True),
            scoring=matt)

        print(nameDataset, name, results)
        dataframe_results.at[nameDataset, name] = results.mean()

dataframe_results.reset_index(inplace=True)
dataframe_results.to_csv("results-unpaired-forest.csv", index=False)