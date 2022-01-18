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

#print("là")

X = np.loadtxt("data/protein_train.data")
y = np.loadtxt("data/protein_train.solution")

X_test = np.loadtxt("data/protein_test.data")
X_valid = np.loadtxt("data/protein_valid.data")

#
scaler = preprocessing.StandardScaler() # .fit(X)
X_scaled = scaler.fit_transform(X)
X_test = scaler.transform(X_test)
X_valid = scaler.transform(X_valid)

# normalizer = preprocessing.Normalizer().fit(X)
# X_normalized = normalizer.transform(X)
# X_normalized = preprocessing.normalize(X, norm='l2')

#
# sel = VarianceThreshold(threshold=(.2 * (1 - .2)))
# sel.fit_transform(X_scaled)

#
# pca = PCA(n_components=20, svd_solver='randomized')
# pca.fit(X_scaled)

#pca = PCA()
# pca = PCA(n_components=200)
# X_final = pca.fit_transform(X_scaled)
# X_test = pca.transform(X_test)
# X_valid = pca.transform(X_valid)
#
# explained_variance = pca.explained_variance_ratio_
# print(explained_variance)

#
X_final = X_scaled

# # penser au 'verbose': [3]
# param_grid_RForestV1 = {'criterion': ['entropy'],
#                      'max_depth': [20],
#                      'max_features': ['sqrt'],
#                      'min_weight_fraction_leaf': [0.0],
#                      'n_estimators': [170],
#                      'n_jobs': [-1]}
#
# param_grid_RForestV2 = {'n_estimators': np.arange(200,401,100),
#                       'criterion':['gini', 'entropy'],
#                       'max_depth':[4,20,50,100,200],
#                       'min_samples_split': np.arange(2,6,2),
#                       'min_samples_leaf': np.arange(2,6,2),
#                       'min_weight_fraction_leaf': np.arange(0,0.6,0.2),
#                       'max_features':['sqrt','log2'],
#                       #'bootstrap':[True, False],
#                       #'oob_score':[True, False],
#                       'n_jobs': [-1]}
#
# param_grid_RForestV3 = {'criterion': ['entropy'],
#                      'max_depth': [20,50,100], #50
#                      'max_features': ['sqrt'],
#                      'min_weight_fraction_leaf': [0.0],
#                      'n_estimators': [200,400,600], #400
#                      'n_jobs': [-1]}
#
# param_grid_RForest = {'criterion': ['entropy'],
#                      'max_depth': [50], #50
#                      'max_features': ['sqrt'],
#                      #'min_weight_fraction_leaf': [0.0],
#                      'n_estimators': [400], #400
#                      'n_jobs': [-1]}
#
#
#
# grid_RForest = GridSearchCV(RandomForestClassifier(), param_grid_RForest, cv=10)
#
# grid_RForest.fit(X_final, y);
#
# grid_RForest.best_params_
# print(grid_RForest.best_params_)
#
# # Predict on the test and validation data.
# y_test = grid_RForest.predict(X_test)
# y_valid = grid_RForest.predict(X_valid)
#
# model_RForest = grid_RForest.best_estimator_
# print(grid_RForest.best_estimator_)
# score_RForest = cross_val_score(model_RForest, X_final, y)
# avg_score_RForest = score_RForest.mean()
# print(avg_score_RForest)
#
# # Save results
# np.savetxt("protein_test.predict", y_test, fmt="%d")
# np.savetxt("protein_valid.predict", y_valid, fmt="%d")
#
# zip_obj = ZipFile('submission.zip', 'w')
# zip_obj.write("protein_test.predict")
# zip_obj.write("protein_valid.predict")
#
# zip_obj.close()

###############################################################################

param_grid_knnV1 = {'n_neighbors': np.arange(1,6),
              'weights':['uniform','distance'],
              'algorithm': ['ball_tree', 'kd_tree', 'brute'],
              'leaf_size': np.arange(10,51,10),
              'p': [1,2],
              'n_jobs': [-1]}

param_grid_knnV2 = {'n_neighbors': [5],
              'weights':['uniform','distance'],
              'algorithm': ['ball_tree', 'kd_tree'],
              'leaf_size': [30],
              'p': [2],
              'n_jobs': [-1]}

param_grid_knnV3 = {'n_neighbors': np.arange(1,10,1),
              'weights':['uniform','distance'],
              'algorithm': ['ball_tree', 'kd_tree'],
              'leaf_size': np.arange(5,51,5),
              'p': [1,2],
              'n_jobs': [-1]}

#{'algorithm': 'ball_tree', 'leaf_size': 5, 'n_jobs': -1, 'n_neighbors': 4, 'p': 2, 'weights': 'distance'}

param_grid_knn = {'n_neighbors': np.arange(1,5,1),
              'weights':['distance'],
              'algorithm': ['ball_tree'],
              'leaf_size': np.arange(1,5,1),
              'p': [2],
              'n_jobs': [-1]}

#{'algorithm': 'ball_tree', 'leaf_size': 1, 'n_jobs': -1, 'n_neighbors': 4, 'p': 2, 'weights': 'distance'}

grid_knn = GridSearchCV(KNeighborsClassifier(), param_grid_knn, cv=10)
grid_knn.fit(X_final, y);
print(grid_knn.best_params_)
model_knn = grid_knn.best_estimator_
score_knn = cross_val_score(model_knn, X, y)
avg_score_knn = score_knn.mean()
print(avg_score_knn)
score_knnBis = cross_val_score(model_knn, X_final, y)
avg_score_knnBis = score_knnBis.mean()
print(avg_score_knnBis)
# Predict on the test and validation data.
y_test = grid_knn.predict(X_test)
y_valid = grid_knn.predict(X_valid)
# Save results
np.savetxt("protein_test_knn10.predict", y_test, fmt="%d")
np.savetxt("protein_valid_knn10.predict", y_valid, fmt="%d")

zip_obj = ZipFile('submission_knn10.zip', 'w')
zip_obj.write("protein_test_knn10.predict")
zip_obj.write("protein_valid_knn10.predict")

zip_obj.close()
