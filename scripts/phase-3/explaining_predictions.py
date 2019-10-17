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
import pickle
import matplotlib.pyplot as plt
import CustomGrid

shap.initjs()

name_dataset = "brca-preproc-filter"

model_to_load = "SVM"

path_to_model = "../binary_models/{0}/all/{1}/{1}".format(name_dataset, model_to_load)
path_to_subset = "../results/{0}/all/{1}.csv".format(name_dataset, model_to_load)

model = pickle.load(open(path_to_model + ".model", "rb")).selected_model
test_indexes = pickle.load(open(path_to_model + ".test_indexes", "rb"))
train_indexes = pickle.load(open(path_to_model + ".train_indexes", "rb"))

# Load the best subset
subsets = pd.read_csv(path_to_subset)
features = subsets.iloc[0, 2].split(' ')

# Load the entire dataset
df = pd.read_csv("../datasets/" + name_dataset + ".csv", index_col=0)

df['Class'] = df['Class'].apply(
        {'N': 0, 'T': 1}.get)

X = df.loc[:, features].to_numpy()
Y = df.loc[:, ['Class']].to_numpy()[:, 0]

transformer = PowerTransformer(
        method='yeo-johnson', standardize=True)

transformer.fit(X[train_indexes])

X_train= transformer.transform(X[train_indexes])
X_test = transformer.transform(X[test_indexes])

Y_train = Y[train_indexes] 
Y_test = Y[test_indexes]

probs = model.predict_proba(X_train)[:,1]
print("Base line value: ", np.mean(probs))

# Create the explainer
explainer = shap.KernelExplainer(model.predict_proba, X_train)

# Explain single predictions from the test set
shap_values = explainer.shap_values(X_test)

# Explain individual predictions
output = shap.force_plot(explainer.expected_value[1], shap_values[1][0,:], X_test[0,:], feature_names= features, matplotlib = False, show=False)

shap.save_html(open("/home/ogreyesp/Desktop/firstPrediction.html", "w"), output)

# Explain individual predictions
output = shap.force_plot(explainer.expected_value[1], shap_values[1][1,:], X_test[1,:], feature_names= features, matplotlib = False, show=False)

shap.save_html(open("/home/ogreyesp/Desktop/secondPrediction.html", "w"), output)

# Plot feature importance
# Explain single predictions from the test set
shap_values = explainer.shap_values(X[train_indexes])
shap.summary_plot(shap_values[1], X[train_indexes], show= False, feature_names= features, class_names= ["Normal", "Tumor"])

plt.savefig("/home/ogreyesp/Desktop/summary_plot1.svg", format="svg")

shap.summary_plot(shap_values[1], X[train_indexes], show= False, feature_names= features, plot_type = "bar", class_names= ["Normal", "Tumor"])

plt.savefig("/home/ogreyesp/Desktop/summary_plot2.svg", format="svg")