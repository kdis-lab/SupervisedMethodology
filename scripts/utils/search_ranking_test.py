
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
np.random.seed(7)
import os

from sklearn.model_selection import cross_validate
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import sklearn
from sklearn.base import clone
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from scipy.stats import wilcoxon

import multiprocessing

# In[2]:


def files_rankings(root, name):
	arr = [os.path.join(root,i) for i in os.listdir(root) if i.startswith(name[:name.rfind('.')] + '-')]
	arr.sort()
	return arr

def new_dataframe(df, columns):
	cols = columns + ['Class']
	return df[cols]

def to_numpy(df, columns):
	new_df = new_dataframe(df,columns)
	data = new_df.to_numpy()
	# ignore class
	X = data[:,:-1]
	X = X.astype(float)
	Y = data[:,-1]
	return X,Y

save_results = lambda results, columns, metric: results.append({
	'NumberAtts':len(columns),
	'Atts':' '.join(columns), 
	'metricOpt':metric}, ignore_index=True)

# eval_model = lambda model, X, y: cross_validate(model, X, y, cv=10, scoring=['roc_auc'], n_jobs=-1)['test_roc_auc'].mean()

def eval_continue(df, model, columns, pFolds=None):
	X, y = to_numpy(df, columns)
	avg_folds, scores, folds = eval_model(model, X, y, pFolds)
	return avg_folds, scores, folds

# In[3]:


def to_binary(x):
	x[x == 'N'] = 0
	x[x == 'T'] = 1
	return x.astype(int)

def train_cv(X,Y):
	y = np.copy(Y)
	y = to_binary(y)
	counts = np.bincount(y)
	paired = counts[0] == counts[1]
	if paired:
		half = X.shape[0]//2
		test = [[i,i+half] for i in range(half)]
		X_in = [i for i in range(X.shape[0])]
		train = [(list(set(X_in) - set(ids)), ids) for ids in test]
	else:
		ind = np.argmin(counts)
		folds = counts[ind]
		kfold = StratifiedKFold(n_splits=folds)
		train = [(t, test) for t, test in kfold.split(X, y)]
	return train

def one_fold(model,X,y,train,test):
	estimator = clone(model)
	estimator.fit(X[train],y[train])
	y_predic = to_binary(estimator.predict(X[test]))
	y_true = to_binary(y[test])
	return roc_auc_score(y_true,y_predic)

def eval_model(model, X, y, pFolds=None): 
	folds = train_cv(X,y) if pFolds is None else pFolds
	with multiprocessing.Pool() as pool:
		scores = pool.starmap(one_fold, [(model,X,y,train,test) for train,test in folds])
	return np.asarray(scores).mean(), np.asarray(scores), folds

# In[5]:


def process_file(pathD, pathRs):
	print(pathD)
	df = pd.read_csv(pathD, index_col=0)
	
	for path_rank in pathRs:
		print(path_rank)
		dfR = pd.read_csv(path_rank, index_col=1)
		indices = dfR.index.values

		for model, m_name in models:
			results = pd.DataFrame(columns=['ID','NumberAtts','Atts','metricOpt'])
			results.set_index('ID', inplace=True)

			# se puso aqui en 1 para limitar el tiempo de busqueda
			for i in range(1):
				columns = [indices[i]]

				general_value, general_scores, folds = eval_continue(df, model, columns, pFolds=None)
				results = save_results(results, columns, general_value)

				if general_value == 1:
					break

				for j in range(i+1, attr_limit + attr_limit):
					new_columns = columns+[indices[j]]

					curr_value, curr_scores, c_folds = eval_continue(df, model, new_columns, pFolds=folds)

					if np.array_equal(general_scores,curr_scores):
						continue
					#if general_value < curr_value:
					if (general_value < curr_value) and (wilcoxon(general_scores,curr_scores,alternative='less').pvalue <= 0.05):
						general_value = curr_value
						columns = new_columns
						results = save_results(results, columns, general_value)
						if general_value == 1:
							break
				if general_value == 1:
					break

			filedir = os.path.join(root, pathD.replace('.csv',''), 
								   path_rank[path_rank.rfind('-')+1:].replace('.csv',''))
			if not os.path.exists(filedir):
				os.makedirs(filedir)
			results.index.name = 'ID'
			results.sort_values(['metricOpt','NumberAtts'],ascending=[False,True],inplace=True)
			results.to_csv(os.path.join(filedir, m_name + '.csv'))
			print('end')

# In[4]:


attr_limit = 50
models = [
	(RandomForestClassifier(n_estimators=100), 'RF'), 
	(LogisticRegression(solver='liblinear'), 'LR'), 
	(SVC(gamma='scale'), 'svm')
]
root = './3'
files = [(os.path.join(root,i), files_rankings(root,i)) for i in os.listdir(root) if i.endswith('-filter.csv')]

# In[6]:


for pathD, pathRs in files:
	process_file(pathD, pathRs)
