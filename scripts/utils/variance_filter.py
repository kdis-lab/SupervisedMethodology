import pandas as pd
from sklearn.feature_selection import VarianceThreshold
import os

root = './datasets'
files = [os.path.join(root,i) for i in os.listdir(root) if i.endswith('-preproc.csv')]

for path in files:
	# use sample name like index
	df = pd.read_csv(path, index_col=0)
	data = df.to_numpy()
	# ignore class
	X = data[:,:-1]
	X = X.astype(float)
	# filter
	selector = VarianceThreshold()
	selector.fit(X)
	# get mask
	selected = selector.get_support(indices=False)
	# verify
	print(path,'was selected',selected[selected == True].shape,'attributes')
	print(path,'was removed',selected[selected == False].shape,'attributes')
	# drop columns
	df.drop(columns=[df.columns[ind] for ind in range(selected.shape[0]) if not selected[ind]], inplace=True)
	df.to_csv(path[:path.rfind('.')] + '-filter.csv')
