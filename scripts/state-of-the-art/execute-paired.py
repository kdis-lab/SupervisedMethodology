from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn import preprocessing
from sklearn import datasets
from sklearn.model_selection import LeaveOneGroupOut
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

from gcready import MyGCForest, get_toy_config
		
np.random.seed(7)

datasets = {
    'COAD': "datasets/coad_gene_read_paired-normal-tumor-filtered.csv",
    'BRCA': "datasets/brca_gene_read_paired-normal-tumor-filtered.csv",
    'HNSC': "datasets/hnsc_gene_read_paired-normal-tumor-filtered.csv",
    'KICH': "datasets/kich_gene_read_paired-normal-tumor-filtered.csv",
    'KIRC': "datasets/kirc_gene_read_paired-normal-tumor-filtered.csv",
    'LUAD': "datasets/luad_gene_read_paired-normal-tumor-filtered.csv",
    'LUSC': "datasets/lusc_gene_read_paired-normal-tumor-filtered.csv",
    'PRAD': "datasets/prad_gene_read_paired-normal-tumor-filtered.csv",
    'THCA': "datasets/thca_gene_read_paired-normal-tumor-filtered.csv"
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

    Y = 1 * (dataset["Class"].values == "T")

    gp = [x for x in range(int(X.shape[0] / 2))]
    gp += gp

    for name, classifier in classifiers.items():

        results = cross_val_score(
            MyGCForest(get_toy_config()), X=X, y=Y, groups=gp, cv=LeaveOneGroupOut(), scoring=matt)
        print(nameDataset, name, classifier, results)
        dataframe_results.at[nameDataset, name] = results.mean()

dataframe_results.reset_index(inplace = True)
dataframe_results.to_csv("results.csv", index=False)
