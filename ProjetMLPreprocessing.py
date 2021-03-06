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
from sklearn.neural_network import MLPClassifier

# from sklearn.preprocessing import MinMaxScaler
# from sklearn.preprocessing import minmax_scale
# from sklearn.preprocessing import MaxAbsScaler
# from sklearn.preprocessing import StandardScaler
# from sklearn.preprocessing import RobustScaler
# from sklearn.preprocessing import Normalizer
# from sklearn.preprocessing import QuantileTransformer
# from sklearn.preprocessing import PowerTransformer

X = np.loadtxt("data/protein_train.data")
y = np.loadtxt("data/protein_train.solution")

X_test = np.loadtxt("data/protein_test.data")
X_valid = np.loadtxt("data/protein_valid.data")

#
scaler = preprocessing.StandardScaler() # .fit(X)
X_standard = scaler.fit_transform(X)
X_test_standard = scaler.transform(X_test)
X_valid_standard = scaler.transform(X_valid)
pca = PCA(n_components=100)
X_standardN = pca.fit_transform(X_standard)
X_test_standardN = pca.transform(X_test_standard)
X_valid_standardN = pca.transform(X_valid_standard)
pca = PCA(n_components=500)
X_standardNBis = pca.fit_transform(X_standard)
X_test_standardNBis = pca.transform(X_test_standard)
X_valid_standardNBis = pca.transform(X_valid_standard)
# pcaR = PCA(n_components=100, svd_solver='randomized')
# X_standardR = pcaR.fit_transform(X_standard)
# X_test_standardR = pcaR.transform(X_test_standard)
# X_valid_standardR = pcaR.transform(X_valid_standard)

# scaler = preprocessing.MinMaxScaler() # .fit(X)
# X_minMax = scaler.fit_transform(X)
# X_test_minMax = scaler.transform(X_test)
# X_valid_minMax = scaler.transform(X_valid)
# pca = PCA(n_components=100)
# X_minMax = pca.fit_transform(X_minMax)
# X_test_minMax = pca.transform(X_test_minMax)
# X_valid_minMax = pca.transform(X_valid_minMax)
#
# scaler = preprocessing.MaxAbsScaler() # .fit(X)
# X_maxAbs = scaler.fit_transform(X)
# X_test_maxAbs = scaler.transform(X_test)
# X_valid_maxAbs = scaler.transform(X_valid)
# pca = PCA(n_components=100)
# X_maxAbs = pca.fit_transform(X_maxAbs)
# X_test_maxAbs = pca.transform(X_test_maxAbs)
# X_valid_maxAbs = pca.transform(X_valid_maxAbs)
#
# scaler = preprocessing.RobustScaler(quantile_range=(25, 75)) # .fit(X)
# X_robust = scaler.fit_transform(X)
# X_test_robust = scaler.transform(X_test)
# X_valid_robust = scaler.transform(X_valid)
# pca = PCA(n_components=100)
# X_robust = pca.fit_transform(X_robust)
# X_test_robust = pca.transform(X_test_robust)
# X_valid_robust = pca.transform(X_valid_robust)

scaler = preprocessing.PowerTransformer(method="yeo-johnson") # .fit(X)
X_powerYJ = scaler.fit_transform(X)
X_test_powerYJ = scaler.transform(X_test)
X_valid_powerYJ = scaler.transform(X_valid)
pca = PCA(n_components=100)
X_powerYJ = pca.fit_transform(X_powerYJ)
X_test_powerYJ = pca.transform(X_test_powerYJ)
X_valid_powerYJ = pca.transform(X_valid_powerYJ)
# pca = PCA(n_components=500)
# X_powerYJBis = pca.fit_transform(X_powerYJ)
# X_test_powerYJBis = pca.transform(X_test_powerYJ)
# X_valid_powerYJBis = pca.transform(X_valid_powerYJ)

# scaler = preprocessing.PowerTransformer(method="box-cox") # .fit(X)
# X_powerBC = scaler.fit_transform(X)
# X_test_powerBC = scaler.transform(X_test)
# X_valid_powerBC = scaler.transform(X_valid)

# scaler = preprocessing.QuantileTransformer(output_distribution="uniform") # .fit(X)
# X_quantileTU = scaler.fit_transform(X)
# X_test_quantileTU = scaler.transform(X_test)
# X_valid_quantileTU = scaler.transform(X_valid)
# pca = PCA(n_components=100)
# X_quantileTU = pca.fit_transform(X_quantileTU)
# X_test_quantileTU = pca.transform(X_test_quantileTU)
# X_valid_quantileTU = pca.transform(X_valid_quantileTU)
#
# scaler = preprocessing.QuantileTransformer(output_distribution="normal") # .fit(X)
# X_quantileTN = scaler.fit_transform(X)
# X_test_quantileTN = scaler.transform(X_test)
# X_valid_quantileTN = scaler.transform(X_valid)
# pca = PCA(n_components=100)
# X_quantileTN = pca.fit_transform(X_quantileTN)
# X_test_quantileTN = pca.transform(X_test_quantileTN)
# X_valid_quantileTN = pca.transform(X_valid_quantileTN)
#
# scaler = preprocessing.Normalizer() # .fit(X)
# X_normal = scaler.fit_transform(X)
# X_test_normal = scaler.transform(X_test)
# X_valid_normal = scaler.transform(X_valid)
# pca = PCA(n_components=100)
# X_normalN = pca.fit_transform(X_normal)
# X_test_normalN = pca.transform(X_test_normal)
# X_valid_normalN = pca.transform(X_valid_normal)
# pcaR = PCA(n_components=100, svd_solver='randomized')
# X_normalR = pcaR.fit_transform(X_normal)
# X_test_normalR = pcaR.transform(X_test_normal)
# X_valid_normalR = pcaR.transform(X_valid_normal)

# normalizer = preprocessing.Normalizer().fit(X)
# X_normalized = normalizer.transform(X)
# X_normalized = preprocessing.normalize(X, norm='l2')

#
# sel = VarianceThreshold(threshold=(.2 * (1 - .2)))
# sel.fit_transform(X_scaled)

#
# pca = PCA(n_components=20, svd_solver='randomized')
# pca.fit(X_scaled)
#
# pca = PCA()
# pca = PCA(n_components=200)
# X_final = pca.fit_transform(X_scaled)
# X_test = pca.transform(X_test)
# X_valid = pca.transform(X_valid)
#
# explained_variance = pca.explained_variance_ratio_
# print(explained_variance)

#
#X_final = X_scaled

# #{'algorithm': 'ball_tree', 'leaf_size': 5, 'n_jobs': -1, 'n_neighbors': 4, 'p': 2, 'weights': 'distance'}
#
# param_grid_knnV4 = {'n_neighbors': np.arange(1,5,1),
#               'weights':['distance'],
#               'algorithm': ['ball_tree'],
#               'leaf_size': np.arange(1,5,1),
#               'p': [2],
#               'n_jobs': [-1]}
#
# #{'algorithm': 'ball_tree', 'leaf_size': 1, 'n_jobs': -1, 'n_neighbors': 4, 'p': 2, 'weights': 'distance'}
#
# param_grid_knn = {'n_neighbors': [4],
#               'weights':['distance'],
#               'algorithm': ['ball_tree'],
#               'leaf_size': [1],
#               'p': [2],
#               'n_jobs': [-1]}

# param_grid_adaBoost = {'algorithm' : ['SAMME','SAMME.R'],
#                        'learning_rate' : [0.5,1.0,1.5],
#                        'n_estimators' : [30,50,80]}

# param_grid_MLP1 = {'hidden_layer_sizes': [(150,100,50)],
#                  'activation': ['identity', 'logistic', 'tanh', 'relu'],
#                  #'alpha': [0.0001,0.0005,0.00005],
#                  'learning_rate': ['constant', 'invscaling', 'adaptive'],
#                  #'learning_rate_init': [0.001,0.005,0.0005],
#                  'solver': ['lbfgs', 'sgd', 'adam'],
#                  'max_iter': [300]}

# param_grid_MLP2 = {'hidden_layer_sizes': [(150,100,50),(300,150,75),(100,50,25)],
#                  'activation': ['relu'],
#                  #'alpha': [0.0001,0.0005,0.00005],
#                  'learning_rate': ['constant'],
#                  #'learning_rate_init': [0.001,0.005,0.0005],
#                  'solver': ['lbfgs'],
#                  'max_iter': [500]}

# {'activation': 'relu', 'hidden_layer_sizes': (150, 100, 50), 'learning_rate': 'constant', 'max_iter': 300, 'solver': 'lbfgs'}

# param_grid_MLP3 = {'hidden_layer_sizes': [(150,100,50),(100,80,20),(100,25),(150,50),(10,10),(10,10,10)],
#                  'activation': ['relu'],
#                  #'alpha': [0.0001,0.0005,0.00005],
#                  'learning_rate': ['constant'],
#                  #'learning_rate_init': [0.001,0.005,0.0005],
#                  'solver': ['lbfgs'],
#                  'max_iter': [500]}

#{'activation': 'relu', 'hidden_layer_sizes': (300, 150, 75), 'learning_rate': 'constant', 'max_iter': 500, 'solver': 'lbfgs'}

# param_grid_MLP4 = {'hidden_layer_sizes': [(300, 150, 75),(500,250,125),(300,300,300),(500,500),(500,500,500),(100,100,100)],
#                  'activation': ['relu'],
#                  #'alpha': [0.0001,0.0005,0.00005],
#                  'learning_rate': ['constant'],
#                  #'learning_rate_init': [0.001,0.005,0.0005],
#                  'solver': ['lbfgs'],
#                  'max_iter': [500]}

# {'activation': 'relu', 'hidden_layer_sizes': (500, 500, 500), 'learning_rate': 'constant', 'max_iter': 500, 'solver': 'lbfgs'}

# param_grid_MLP5 = {'hidden_layer_sizes': [(100),(100,100),(100,100,100),(100,100,100,100,100)],
#                  'activation': ['relu'],
#                  #'alpha': [0.0001,0.0005,0.00005],
#                  'learning_rate': ['constant'],
#                  #'learning_rate_init': [0.001,0.005,0.0005],
#                  'solver': ['lbfgs'],
#                  'max_iter': [500]}

param_grid_MLP = {'hidden_layer_sizes': [(1000,1000,1000)],
                 'activation': ['relu'],
                 #'alpha': [0.0001,0.0005,0.00005],
                 'learning_rate': ['constant'],
                 #'learning_rate_init': [0.001,0.005,0.0005],
                 'solver': ['lbfgs'],
                 'max_iter': [500]}

#grid_adaBoost = GridSearchCV(KNeighborsClassifier(), param_grid_adaBoost, cv=10)
#grid_adaBoost = GridSearchCV(AdaBoostClassifier(), param_grid_adaBoost, cv=10)
grid_MLP = GridSearchCV(MLPClassifier(), param_grid_MLP, cv=10, n_jobs=-1)
grid_MLP.fit(X_standardN, y);
print(grid_MLP.best_params_)
model_MLP = grid_MLP.best_estimator_
score_MLP = cross_val_score(model_MLP, X, y)
avg_score_MLP = score_MLP.mean()
print(avg_score_MLP)
score_MLPBis = cross_val_score(model_MLP, X_standardN, y)
avg_score_MLPBis = score_MLPBis.mean()
print(avg_score_MLPBis)
# Predict on the test and validation data.
y_test = grid_MLP.predict(X_test_standardN)
y_valid = grid_MLP.predict(X_valid_standardN)
# Save results
np.savetxt("protein_test_MLPStandard100Bis.predict", y_test, fmt="%d")
np.savetxt("protein_valid_MLPStandard100Bis.predict", y_valid, fmt="%d")

zip_obj = ZipFile('submission_MLPStandard100Bis.zip', 'w')
zip_obj.write("protein_test_MLPStandard100Bis.predict")
zip_obj.write("protein_valid_MLPStandard100Bis.predict")

zip_obj.close()


grid_MLP.fit(X_standardNBis, y);
print(grid_MLP.best_params_)
model_MLP = grid_MLP.best_estimator_
score_MLP = cross_val_score(model_MLP, X, y)
avg_score_MLP = score_MLP.mean()
print(avg_score_MLP)
score_MLPBis = cross_val_score(model_MLP, X_standardNBis, y)
avg_score_MLPBis = score_MLPBis.mean()
print(avg_score_MLPBis)
# Predict on the test and validation data.
y_test = grid_MLP.predict(X_test_standardNBis)
y_valid = grid_MLP.predict(X_valid_standardNBis)
# Save results
np.savetxt("protein_test_MLPStandard500Bis.predict", y_test, fmt="%d")
np.savetxt("protein_valid_MLPStandard500Bis.predict", y_valid, fmt="%d")

zip_obj = ZipFile('submission_MLPStandard500Bis.zip', 'w')
zip_obj.write("protein_test_MLPStandard500Bis.predict")
zip_obj.write("protein_valid_MLPStandard500Bis.predict")

zip_obj.close()

# #
# grid_MLP = GridSearchCV(KNeighborsClassifier(), param_grid_MLP, cv=10)
# grid_MLP.fit(X_standardR, y);
# print(grid_MLP.best_params_)
# model_MLP = grid_MLP.best_estimator_
# score_MLP = cross_val_score(model_MLP, X, y)
# avg_score_MLP = score_MLP.mean()
# print(avg_score_MLP)
# score_MLPBis = cross_val_score(model_MLP, X_standardR, y)
# avg_score_MLPBis = score_MLPBis.mean()
# print(avg_score_MLPBis)
#
# grid_MLP.fit(X_minMax, y);
# print(grid_MLP.best_params_)
# model_MLP = grid_MLP.best_estimator_
# score_MLP = cross_val_score(model_MLP, X, y)
# avg_score_MLP = score_MLP.mean()
# print(avg_score_MLP)
# score_MLPBis = cross_val_score(model_MLP, X_minMax, y)
# avg_score_MLPBis = score_MLPBis.mean()
# print(avg_score_MLPBis)
# # Predict on the test and validation data.
# y_test = grid_MLP.predict(X_test_minMax)
# y_valid = grid_MLP.predict(X_valid_minMax)
# # Save results
# np.savetxt("protein_test_MLPMinMax4.predict", y_test, fmt="%d")
# np.savetxt("protein_valid_MLPMinMax4.predict", y_valid, fmt="%d")
#
# zip_obj = ZipFile('submission_MLPMinMax4.zip', 'w')
# zip_obj.write("protein_test_MLPMinMax4.predict")
# zip_obj.write("protein_valid_MLPMinMax4.predict")

# zip_obj.close()
#
# grid_MLP.fit(X_maxAbs, y);
# print(grid_MLP.best_params_)
# model_MLP = grid_MLP.best_estimator_
# score_MLP = cross_val_score(model_MLP, X, y)
# avg_score_MLP = score_MLP.mean()
# print(avg_score_MLP)
# score_MLPBis = cross_val_score(model_MLP, X_maxAbs, y)
# avg_score_MLPBis = score_MLPBis.mean()
# print(avg_score_MLPBis)
# # Predict on the test and validation data.
# y_test = grid_MLP.predict(X_test_maxAbs)
# y_valid = grid_MLP.predict(X_valid_maxAbs)
# # Save results
# np.savetxt("protein_test_MLPMaxAbs4.predict", y_test, fmt="%d")
# np.savetxt("protein_valid_MLPMaxAbs4.predict", y_valid, fmt="%d")
#
# zip_obj = ZipFile('submission_MLPMaxAbs4.zip', 'w')
# zip_obj.write("protein_test_MLPMaxAbs4.predict")
# zip_obj.write("protein_valid_MLPMaxAbs4.predict")
#
# zip_obj.close()
#
# grid_MLP.fit(X_robust, y);
# print(grid_MLP.best_params_)
# model_MLP = grid_MLP.best_estimator_
# score_MLP = cross_val_score(model_MLP, X, y)
# avg_score_MLP = score_MLP.mean()
# print(avg_score_MLP)
# score_MLPBis = cross_val_score(model_MLP, X_robust, y)
# avg_score_MLPBis = score_MLPBis.mean()
# print(avg_score_MLPBis)
# # Predict on the test and validation data.
# y_test = grid_MLP.predict(X_test_robust)
# y_valid = grid_MLP.predict(X_valid_robust)
# # Save results
# np.savetxt("protein_test_MLPRobust4.predict", y_test, fmt="%d")
# np.savetxt("protein_valid_MLPRobust4.predict", y_valid, fmt="%d")
#
# zip_obj = ZipFile('submission_MLPRobust4.zip', 'w')
# zip_obj.write("protein_test_MLPRobust4.predict")
# zip_obj.write("protein_valid_MLPRobust4.predict")
#
# zip_obj.close()
#
grid_MLP.fit(X_powerYJ, y);
print(grid_MLP.best_params_)
model_MLP = grid_MLP.best_estimator_
score_MLP = cross_val_score(model_MLP, X, y)
avg_score_MLP = score_MLP.mean()
print(avg_score_MLP)
score_MLPBis = cross_val_score(model_MLP, X_powerYJ, y)
avg_score_MLPBis = score_MLPBis.mean()
print(avg_score_MLPBis)
# Predict on the test and validation data.
y_test = grid_MLP.predict(X_test_powerYJ)
y_valid = grid_MLP.predict(X_valid_powerYJ)
# Save results
np.savetxt("protein_test_MLPPowerYJ100Bis.predict", y_test, fmt="%d")
np.savetxt("protein_valid_MLPPowerYJ100Bis.predict", y_valid, fmt="%d")

zip_obj = ZipFile('submission_MLPPowerYJ100Bis.zip', 'w')
zip_obj.write("protein_test_MLPPowerYJ100Bis.predict")
zip_obj.write("protein_valid_MLPPowerYJ100Bis.predict")

zip_obj.close()

# grid_MLP.fit(X_powerYJBis, y);
# print(grid_MLP.best_params_)
# model_MLP = grid_MLP.best_estimator_
# score_MLP = cross_val_score(model_MLP, X, y)
# avg_score_MLP = score_MLP.mean()
# print(avg_score_MLP)
# score_MLPBis = cross_val_score(model_MLP, X_powerYJBis, y)
# avg_score_MLPBis = score_MLPBis.mean()
# print(avg_score_MLPBis)
# # Predict on the test and validation data.
# y_test = grid_MLP.predict(X_test_powerYJBis)
# y_valid = grid_MLP.predict(X_valid_powerYJBis)
# # Save results
# np.savetxt("protein_test_MLPPowerYJ500.predict", y_test, fmt="%d")
# np.savetxt("protein_valid_MLPPowerYJ500.predict", y_valid, fmt="%d")
#
# zip_obj = ZipFile('submission_MLPPowerYJ500.zip', 'w')
# zip_obj.write("protein_test_MLPPowerYJ500.predict")
# zip_obj.write("protein_valid_MLPPowerYJ500.predict")
#
# zip_obj.close()
#
# grid_MLP.fit(X_powerBC, y);
# print(grid_MLP.best_params_)
# model_MLP = grid_MLP.best_estimator_
# score_MLP = cross_val_score(model_MLP, X, y)
# avg_score_MLP = score_MLP.mean()
# print(avg_score_MLP)
# score_MLPBis = cross_val_score(model_MLP, X_powerBC, y)
# avg_score_MLPBis = score_MLPBis.mean()
# print(avg_score_MLPBis)
#
# grid_MLP.fit(X_quantileTU, y);
# print(grid_MLP.best_params_)
# model_MLP = grid_MLP.best_estimator_
# score_MLP = cross_val_score(model_MLP, X, y)
# avg_score_MLP = score_MLP.mean()
# print(avg_score_MLP)
# score_MLPBis = cross_val_score(model_MLP, X_quantileTU, y)
# avg_score_MLPBis = score_MLPBis.mean()
# print(avg_score_MLPBis)
# # Predict on the test and validation data.
# y_test = grid_MLP.predict(X_test_quantileTU)
# y_valid = grid_MLP.predict(X_valid_quantileTU)
# # Save results
# np.savetxt("protein_test_MLPQuantileTU4.predict", y_test, fmt="%d")
# np.savetxt("protein_valid_MLPQuantileTU4.predict", y_valid, fmt="%d")
#
# zip_obj = ZipFile('submission_MLPQuantileTU4.zip', 'w')
# zip_obj.write("protein_test_MLPQuantileTU4.predict")
# zip_obj.write("protein_valid_MLPQuantileTU4.predict")
#
# zip_obj.close()
# #
# grid_MLP.fit(X_quantileTN, y);
# print(grid_MLP.best_params_)
# model_MLP = grid_MLP.best_estimator_
# score_MLP = cross_val_score(model_MLP, X, y)
# avg_score_MLP = score_MLP.mean()
# print(avg_score_MLP)
# score_MLPBis = cross_val_score(model_MLP, X_quantileTN, y)
# avg_score_MLPBis = score_MLPBis.mean()
# print(avg_score_MLPBis)
# # Predict on the test and validation data.
# y_test = grid_MLP.predict(X_test_quantileTN)
# y_valid = grid_MLP.predict(X_valid_quantileTN)
# # Save results
# np.savetxt("protein_test_MLPQuantileTN4.predict", y_test, fmt="%d")
# np.savetxt("protein_valid_MLPQuantileTN4.predict", y_valid, fmt="%d")
#
# zip_obj = ZipFile('submission_MLPQuantileTN4.zip', 'w')
# zip_obj.write("protein_test_MLPQuantileTN4.predict")
# zip_obj.write("protein_valid_MLPQuantileTN4.predict")
#
# zip_obj.close()
# #
# grid_MLP.fit(X_normalN, y);
# print(grid_MLP.best_params_)
# model_MLP = grid_MLP.best_estimator_
# score_MLP = cross_val_score(model_MLP, X, y)
# avg_score_MLP = score_MLP.mean()
# print(avg_score_MLP)
# score_MLPBis = cross_val_score(model_MLP, X_normalN, y)
# avg_score_MLPBis = score_MLPBis.mean()
# print(avg_score_MLPBis)
# # Predict on the test and validation data.
# y_test = grid_MLP.predict(X_test_normalN)
# y_valid = grid_MLP.predict(X_valid_normalN)
# # Save results
# np.savetxt("protein_test_MLPNormalN4.predict", y_test, fmt="%d")
# np.savetxt("protein_valid_MLPNormalN4.predict", y_valid, fmt="%d")
#
# zip_obj = ZipFile('submission_MLPNormalN4.zip', 'w')
# zip_obj.write("protein_test_MLPNormalN4.predict")
# zip_obj.write("protein_valid_MLPNormalN4.predict")
#
# zip_obj.close()
# #
# grid_MLP.fit(X_normalR, y);
# print(grid_MLP.best_params_)
# model_MLP = grid_MLP.best_estimator_
# score_MLP = cross_val_score(model_MLP, X, y)
# avg_score_MLP = score_MLP.mean()
# print(avg_score_MLP)
# score_MLPBis = cross_val_score(model_MLP, X_normalR, y)
# avg_score_MLPBis = score_MLPBis.mean()
# print(avg_score_MLPBis)


# # Predict on the test and validation data.
# y_test = grid_MLP.predict(X_test)
# y_valid = grid_MLP.predict(X_valid)
# # Save results
# np.savetxt("protein_test_MLP10.predict", y_test, fmt="%d")
# np.savetxt("protein_valid_MLP10.predict", y_valid, fmt="%d")
#
# zip_obj = ZipFile('submission_MLP10.zip', 'w')
# zip_obj.write("protein_test_MLP10.predict")
# zip_obj.write("protein_valid_MLP10.predict")
#
# zip_obj.close()
