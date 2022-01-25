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
from sklearn.svm import NuSVC
from sklearn.svm import LinearSVC

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
pca = PCA(n_components=100)
X_final = pca.fit_transform(X_scaled)
X_test = pca.transform(X_test)
X_valid = pca.transform(X_valid)

explained_variance = pca.explained_variance_ratio_
print(explained_variance)

# param_grid_SVC1 = {#'C': np.arange(0.1,2,0.3),
#                  'kernel': ['linear', 'poly', 'rbf', 'sigmoid']} #, et , 'precomputed'
#                  #'degree': [1,2,3,4,5],
#                  #'gamma': ['scale','auto']}

# param_grid_SVC2 = {'C': np.arange(0.1,2,0.3),
#                  'kernel': ['rbf'],
#                  #'degree': [1,2,3,4,5],
#                  'gamma': ['scale','auto']}

#{'C': 1.9000000000000004, 'gamma': 'auto', 'kernel': 'rbf'}

# param_grid_SVC = {'C': np.arange(1.9,3.0,0.3),
#                  'kernel': ['rbf'],
#                  #'degree': [1,2,3,4,5],
#                  'gamma': ['auto']}

# param_grid_knn = {'n_neighbors': np.arange(1,10,1),
#               'weights':['uniform','distance'],
#               'algorithm': ['ball_tree', 'kd_tree'],
#               'leaf_size': np.arange(5,51,5),
#               'p': [1,2],
#               'n_jobs': [-1]}

param_grid_SVC = {#'C': np.arange(1.9,3.0,0.3),
                 'kernel': ['rbf'],
                 #'degree': [1,2,3,4,5],
                 'gamma': ['auto']}

# param_grid_NuSVC = {#'C': np.arange(1.9,3.0,0.3),
#                  'kernel': ['rbf'],
#                  #'degree': [1,2,3,4,5],
#                  'gamma': ['auto']}

param_grid_LinearSVC = {#'C': np.arange(1.9,3.0,0.3),
                 #'loss': ['hinge','squared_hinge'],
                 'penalty' : ['l1','l2'],
                 'max_iter' : [10000]}


# grid_SVC = GridSearchCV(SVC(), param_grid_SVC, cv=10, n_jobs=-1)
# grid_SVC.fit(X_final, y);
# print(grid_SVC.best_params_)
# model_SVC = grid_SVC.best_estimator_
# score_SVC = cross_val_score(model_SVC, X, y)
# avg_score_SVC = score_SVC.mean()
# print(avg_score_SVC)
# score_SVCBis = cross_val_score(model_SVC, X_final, y)
# avg_score_SVCBis = score_SVCBis.mean()
# print(avg_score_SVCBis)
# # Predict on the test and validation data.
# y_test = grid_SVC.predict(X_test)
# y_valid = grid_SVC.predict(X_valid)
# # Save results
# np.savetxt("protein_test_SVC100.predict", y_test, fmt="%d")
# np.savetxt("protein_valid_SVC100.predict", y_valid, fmt="%d")
#
# zip_obj = ZipFile('submission_SVC100.zip', 'w')
# zip_obj.write("protein_test_SVC100.predict")
# zip_obj.write("protein_valid_SVC100.predict")
#
# zip_obj.close()

# grid_SVC = GridSearchCV(NuSVC(), param_grid_NuSVC, cv=10, n_jobs=-1)
# grid_SVC.fit(X_final, y);
# print(grid_SVC.best_params_)
# model_SVC = grid_SVC.best_estimator_
# score_SVC = cross_val_score(model_SVC, X, y)
# avg_score_SVC = score_SVC.mean()
# print(avg_score_SVC)
# score_SVCBis = cross_val_score(model_SVC, X_final, y)
# avg_score_SVCBis = score_SVCBis.mean()
# print(avg_score_SVCBis)
# # Predict on the test and validation data.
# y_test = grid_SVC.predict(X_test)
# y_valid = grid_SVC.predict(X_valid)
# # Save results
# np.savetxt("protein_test_NuSVC100.predict", y_test, fmt="%d")
# np.savetxt("protein_valid_NuSVC100.predict", y_valid, fmt="%d")
#
# zip_obj = ZipFile('submission_NuSVC100.zip', 'w')
# zip_obj.write("protein_test_NuSVC100.predict")
# zip_obj.write("protein_valid_NuSVC100.predict")
#
# zip_obj.close()

grid_SVC = GridSearchCV(LinearSVC(), param_grid_LinearSVC, cv=10, n_jobs=-1)
grid_SVC.fit(X_final, y);
print(grid_SVC.best_params_)
model_SVC = grid_SVC.best_estimator_
score_SVC = cross_val_score(model_SVC, X, y)
avg_score_SVC = score_SVC.mean()
print(avg_score_SVC)
score_SVCBis = cross_val_score(model_SVC, X_final, y)
avg_score_SVCBis = score_SVCBis.mean()
print(avg_score_SVCBis)
# Predict on the test and validation data.
y_test = grid_SVC.predict(X_test)
y_valid = grid_SVC.predict(X_valid)
# Save results
np.savetxt("protein_test_LinearSVC100.predict", y_test, fmt="%d")
np.savetxt("protein_valid_LinearSVC100.predict", y_valid, fmt="%d")

zip_obj = ZipFile('submission_LinearSVC100.zip', 'w')
zip_obj.write("protein_test_LinearSVC100.predict")
zip_obj.write("protein_valid_LinearSVC100.predict")

zip_obj.close()
