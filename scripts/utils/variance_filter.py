import pandas as pd
from sklearn.feature_selection import VarianceThreshold
import os
import numpy as np

root = './datasets/Imbalanced'
files = [os.path.join(root, i)
         for i in os.listdir(root) if i.endswith('-preproc.csv')]

for path in files:
    
    # use sample name like index
    df = pd.read_csv(path, index_col=0)
    data = df.to_numpy()
    # ignore class
    X = data[:, :-1]
    X = X.astype(float)
    # filter
    selector = VarianceThreshold()
    selector.fit(X)
    # get mask
    selected = selector.get_support(indices=False)
    # verify
    to_retain= selected.sum()
    print(path, 'was selected', to_retain, 'attributes')
    print(path, 'was removed', len(selected) - to_retain, 'attributes')
    
	# drop columns
    df.drop(columns= df.columns.values[:-1][np.logical_not(selected)], inplace=True)

    df.to_csv(path[:path.rfind('.')] + '-filter.csv')