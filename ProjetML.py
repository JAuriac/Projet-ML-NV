import numpy as np
from zipfile import ZipFile
from sklearn import linear_model


from sklearn import metrics
import pandas as pd
#import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score

from sklearn import preprocessing
from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import PCA

print("là")

X = np.loadtxt("data/protein_train.data")
y = np.loadtxt("data/protein_train.solution")

X_test = np.loadtxt("data/protein_test.data")
X_valid = np.loadtxt("data/protein_valid.data")

#
scaler = preprocessing.StandardScaler().fit(X)
X_scaled = scaler.transform(X)

#
# sel = VarianceThreshold(threshold=(.2 * (1 - .2)))
# sel.fit_transform(X_scaled)

pca = PCA(n_components=200, svd_solver='randomized')
pca.fit(X_scaled)

#
X_final = X_scaled

#
param_grid_RForestV1 = {'criterion': ['entropy'],
                     'max_depth': [20],
                     'max_features': ['sqrt'],
                     'min_weight_fraction_leaf': [0.0],
                     'n_estimators': [170],
                     'n_jobs': [-1]}

param_grid_RForest = {'n_estimators': np.arange(200,501,40),
                      'criterion':['gini', 'entropy'],
                      'max_depth':[4,5,6,10,20,50,100,150,200],
                      'min_samples_split': np.arange(1,6,1),
                      'min_samples_leaf': np.arange(1,6,1),
                      'min_weight_fraction_leaf': np.arange(0,0.6,0.1),
                      'max_features':['sqrt','log2'],
                      #'bootstrap':[True, False],
                      #'oob_score':[True, False],
                      'n_jobs': [-1]}

grid_RForest = GridSearchCV(RandomForestClassifier(), param_grid_RForest, cv=10)

grid_RForest.fit(X_final, y);

grid_RForest.best_params_

# Predict on the test and validation data.
y_test = grid_RForest.predict(X_test)
y_valid = grid_RForest.predict(X_valid)

model_RForest = grid_RForest.best_estimator_
score_RForest = cross_val_score(model_RForest, X_final, y)
avg_score_RForest = score_RForest.mean()
print(avg_score_RForest)

# Save results
np.savetxt("protein_test.predict", y_test, fmt="%d")
np.savetxt("protein_valid.predict", y_valid, fmt="%d")

zip_obj = ZipFile('submission.zip', 'w')
zip_obj.write("protein_test.predict")
zip_obj.write("protein_valid.predict")

zip_obj.close()
