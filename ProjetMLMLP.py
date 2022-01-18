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
from sklearn.neural_network import MLPClassifier

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

param_grid_MLP = {'hidden_layer_sizes': [(5, 2),(1, 1),(15,)],
                 'activation': ['identity', 'logistic', 'tanh', 'relu'],
                 #'alpha': [0.0001,0.0005,0.00005],
                 'learning_rate': ['constant', 'invscaling', 'adaptive'],
                 #'learning_rate_init': [0.001,0.005,0.0005],
                 'solver': ['lbfgs', 'sgd', 'adam']}

# param_grid_knn = {'n_neighbors': np.arange(1,10,1),
#               'weights':['uniform','distance'],
#               'algorithm': ['ball_tree', 'kd_tree'],
#               'leaf_size': np.arange(5,51,5),
#               'p': [1,2],
#               'n_jobs': [-1]}


grid_MLP = GridSearchCV(MLPClassifier(), param_grid_MLP, cv=10, n_jobs=-1)
grid_MLP.fit(X_final, y);
print(grid_MLP.best_params_)
model_MLP = grid_MLP.best_estimator_
score_MLP = cross_val_score(model_MLP, X, y)
avg_score_MLP = score_MLP.mean()
print(avg_score_MLP)
score_MLPBis = cross_val_score(model_MLP, X_final, y)
avg_score_MLPBis = score_MLPBis.mean()
print(avg_score_MLPBis)
# Predict on the test and validation data.
y_test = grid_MLP.predict(X_test)
y_valid = grid_MLP.predict(X_valid)
# Save results
np.savetxt("protein_test_MLP1.predict", y_test, fmt="%d")
np.savetxt("protein_valid_MLP1.predict", y_valid, fmt="%d")

zip_obj = ZipFile('submission_MLP1.zip', 'w')
zip_obj.write("protein_test_MLP1.predict")
zip_obj.write("protein_valid_MLP1.predict")

zip_obj.close()
