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

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.mixture import BayesianGaussianMixture

X = np.loadtxt("data/protein_train.data")
y = np.loadtxt("data/protein_train.solution")

X_test = np.loadtxt("data/protein_test.data")
X_valid = np.loadtxt("data/protein_valid.data")

#
scaler = preprocessing.StandardScaler() # .fit(X)
X_scaled = scaler.fit_transform(X)
X_test = scaler.transform(X_test)
X_valid = scaler.transform(X_valid)

#
pca = PCA(n_components=12)
X_final = pca.fit_transform(X_scaled)
X_test = pca.transform(X_test)
X_valid = pca.transform(X_valid)

explained_variance = pca.explained_variance_ratio_
print(explained_variance)

param_grid_GBM = {'n_components': np.arange(1,4,1),
                 'covariance_type': ['full', 'tied', 'diag', 'spherical'],
                 'tol': [1e-3,1e-5],
                 'max_iter': [50,100,150],
                 'n_init': [1,2,3],
                 'init_params': ['kmeans', 'random']}

# param_grid_knn = {'n_neighbors': np.arange(1,10,1),
#               'weights':['uniform','distance'],
#               'algorithm': ['ball_tree', 'kd_tree'],
#               'leaf_size': np.arange(5,51,5),
#               'p': [1,2],
#               'n_jobs': [-1]}


grid_GBM = GridSearchCV(BayesianGaussianMixture(), param_grid_GBM, cv=10, n_jobs=-1)
grid_GBM.fit(X_final, y);
print(grid_GBM.best_params_)
model_GBM = grid_GBM.best_estimator_
score_GBM = cross_val_score(model_GBM, X, y)
avg_score_GBM = score_GBM.mean()
print(avg_score_GBM)
score_GBMBis = cross_val_score(model_GBM, X_final, y)
avg_score_GBMBis = score_GBMBis.mean()
print(avg_score_GBMBis)
# Predict on the test and validation data.
y_test = grid_GBM.predict(X_test)
y_valid = grid_GBM.predict(X_valid)
# Save results
np.savetxt("protein_test_GBM1.predict", y_test, fmt="%d")
np.savetxt("protein_valid_GBM1.predict", y_valid, fmt="%d")

zip_obj = ZipFile('submission_GBM1.zip', 'w')
zip_obj.write("protein_test_GBM1.predict")
zip_obj.write("protein_valid_GBM1.predict")

zip_obj.close()
