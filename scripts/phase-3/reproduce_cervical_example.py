import sklearn
from sklearn.model_selection import train_test_split
import numpy as np
import shap
import time
import pandas as pd
from sklearn.preprocessing import PowerTransformer
from sklearn.metrics import make_scorer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

#shap.initjs()

cerv = pd.read_csv("datasets/cervical.csv")

y = cerv["Biopsy"].values == "Cancer"
X = cerv.drop(["Biopsy"], axis=1)

mod = RandomForestClassifier(n_estimators = 100, random_state = 42)
mod.fit(X,y)

explainer = shap.TreeExplainer(mod, X, model_output = "margin")
shap_values = explainer.shap_values(X)

x = mod.predict_proba(X)[:,1]
np.mean(x)

i = 18

# Explain individual predictions
shap.force_plot(explainer.expected_value[1], shap_values[1][i,:], X.iloc[i,:], matplotlib = True, show=False)

i = 6

# Explain individual predictions
shap.force_plot(explainer.expected_value[1], shap_values[1][i,:], X.iloc[i,:], matplotlib = True, show=False)

# Plot feature importance
shap.summary_plot(shap_values[1], X,  show=False)

plt.savefig("/home/ogreyesp/Desktop/summary_plot.eps", format="eps")

