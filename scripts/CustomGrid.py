import itertools as it
import sklearn
import numpy as np
from joblib import Parallel, delayed

class CustomGrid:

    def __init__(self, model, parameters, cv=3):
        self.selected_model = None
        self.model = model
        self.cv = cv
        self.parameters = parameters
        keys = sorted(parameters)
        combinations = it.product(*(parameters[k] for k in keys))
        combinations = list(combinations)
        self.parameters_combination = [{keys[i]: param[i] for i in range(len(keys))} for param in combinations ]

    def fit(self, x, y):
        scores = Parallel(n_jobs=20)(delayed(self.evaluate)(self.model,param,x,y) for param in self.parameters_combination)
        scores = np.asarray(scores)
        best = np.argmax(scores)
        best_params, best_score = self.parameters_combination[best], scores[best]

        self.selected_model = sklearn.base.clone(self.model)
        self.selected_model.set_params(**best_params)
        self.selected_model.fit(x,y)

    def predict(self, x):
        return self.selected_model.predict(x)

    def evaluate(self,model,param,x,y):
        print('eval', param)
        scores = []
        kfold = sklearn.model_selection.StratifiedKFold(self.cv,shuffle=True)
        for train, test in kfold.split(x, y):
            model_fold = sklearn.base.clone(model)
            model_fold.set_params(**param)
            model_fold.fit(x[train], y[train])

            y_pred = model_fold.predict(x[test])
            score = sklearn.metrics.roc_auc_score(y[test], y_pred, average='macro')
            scores.append(score)
        return np.asarray(scores).mean()

# /////////////////////////
#from sklearn.svm import SVC
#import numpy as np

#parameters = {
#    # types of kernels to be tested
#    "kernel": ["linear", "poly", "rbf", "sigmoid"],
#    "C": [0.01, 0.1, 1, 10],  # range of C to be tested
#    "degree": [1, 2, 3]  # degrees to be tested
#}
#grid = CustomGrid(SVC(gamma="auto"), parameters)        
#X = np.random.randint(5, size=(10000, 4))
#y = np.random.randint(2, size=(10000))
#grid.fit(X,y)

#grid.predict(X)

#print(sklearn.metrics.roc_auc_score(y,grid.predict(X)))