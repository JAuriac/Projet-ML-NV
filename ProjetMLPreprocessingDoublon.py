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
from sklearn.svm import SVC

from sklearn.neighbors import KNeighborsClassifier

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
# pcaR = PCA(n_components=100, svd_solver='randomized')
# X_standardR = pcaR.fit_transform(X_standard)
# X_test_standardR = pcaR.transform(X_test_standard)
# X_valid_standardR = pcaR.transform(X_valid_standard)

scaler = preprocessing.MinMaxScaler() # .fit(X)
X_minMax = scaler.fit_transform(X)
X_test_minMax = scaler.transform(X_test)
X_valid_minMax = scaler.transform(X_valid)
pca = PCA(n_components=100)
X_minMax = pca.fit_transform(X_minMax)
X_test_minMax = pca.transform(X_test_minMax)
X_valid_minMax = pca.transform(X_valid_minMax)

scaler = preprocessing.MaxAbsScaler() # .fit(X)
X_maxAbs = scaler.fit_transform(X)
X_test_maxAbs = scaler.transform(X_test)
X_valid_maxAbs = scaler.transform(X_valid)
pca = PCA(n_components=100)
X_maxAbs = pca.fit_transform(X_maxAbs)
X_test_maxAbs = pca.transform(X_test_maxAbs)
X_valid_maxAbs = pca.transform(X_valid_maxAbs)

scaler = preprocessing.RobustScaler(quantile_range=(25, 75)) # .fit(X)
X_robust = scaler.fit_transform(X)
X_test_robust = scaler.transform(X_test)
X_valid_robust = scaler.transform(X_valid)
pca = PCA(n_components=100)
X_robust = pca.fit_transform(X_robust)
X_test_robust = pca.transform(X_test_robust)
X_valid_robust = pca.transform(X_valid_robust)

scaler = preprocessing.PowerTransformer(method="yeo-johnson") # .fit(X)
X_powerYJ = scaler.fit_transform(X)
X_test_powerYJ = scaler.transform(X_test)
X_valid_powerYJ = scaler.transform(X_valid)
pca = PCA(n_components=100)
X_powerYJ = pca.fit_transform(X_powerYJ)
X_test_powerYJ = pca.transform(X_test_powerYJ)
X_valid_powerYJ = pca.transform(X_valid_powerYJ)

# scaler = preprocessing.PowerTransformer(method="box-cox") # .fit(X)
# X_powerBC = scaler.fit_transform(X)
# X_test_powerBC = scaler.transform(X_test)
# X_valid_powerBC = scaler.transform(X_valid)

scaler = preprocessing.QuantileTransformer(output_distribution="uniform") # .fit(X)
X_quantileTU = scaler.fit_transform(X)
X_test_quantileTU = scaler.transform(X_test)
X_valid_quantileTU = scaler.transform(X_valid)
pca = PCA(n_components=100)
X_quantileTU = pca.fit_transform(X_quantileTU)
X_test_quantileTU = pca.transform(X_test_quantileTU)
X_valid_quantileTU = pca.transform(X_valid_quantileTU)

scaler = preprocessing.QuantileTransformer(output_distribution="normal") # .fit(X)
X_quantileTN = scaler.fit_transform(X)
X_test_quantileTN = scaler.transform(X_test)
X_valid_quantileTN = scaler.transform(X_valid)
pca = PCA(n_components=100)
X_quantileTN = pca.fit_transform(X_quantileTN)
X_test_quantileTN = pca.transform(X_test_quantileTN)
X_valid_quantileTN = pca.transform(X_valid_quantileTN)

scaler = preprocessing.Normalizer() # .fit(X)
X_normal = scaler.fit_transform(X)
X_test_normal = scaler.transform(X_test)
X_valid_normal = scaler.transform(X_valid)
pca = PCA(n_components=100)
X_normalN = pca.fit_transform(X_normal)
X_test_normalN = pca.transform(X_test_normal)
X_valid_normalN = pca.transform(X_valid_normal)
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
#                        'n_estimators' : [30,50,80,150]}

param_grid_SVC = {'algorithm' : ['SAMME','SAMME.R'],
                 'learning_rate' : [0.5,1.0,1.5],
                 'n_estimators' : [30,50,80,150]}

#grid_knn = GridSearchCV(KNeighborsClassifier(), param_grid_knn, cv=10)
grid_svc = GridSearchCV(SVC(), param_grid_svc, cv=10)
grid_svc.fit(X_standardN, y);
print(grid_svc.best_params_)
model_svc = grid_svc.best_estimator_
score_svc = cross_val_score(model_svc, X, y)
avg_score_svc = score_svc.mean()
print(avg_score_svc)
score_svcBis = cross_val_score(model_svc, X_standardN, y)
avg_score_svcBis = score_svcBis.mean()
print(avg_score_svcBis)
# Predict on the test and validation data.
y_test = grid_svc.predict(X_test_standardN)
y_valid = grid_svc.predict(X_valid_standardN)
# Save results
np.savetxt("protein_test_svcStandard.predict", y_test, fmt="%d")
np.savetxt("protein_valid_svcStandard.predict", y_valid, fmt="%d")

zip_obj = ZipFile('submission_svcStandard.zip', 'w')
zip_obj.write("protein_test_svcStandard.predict")
zip_obj.write("protein_valid_svcStandard.predict")

zip_obj.close()
# #
# grid_svc = GridSearchCV(KNeighborsClassifier(), param_grid_svc, cv=10)
# grid_svc.fit(X_standardR, y);
# print(grid_svc.best_params_)
# model_svc = grid_svc.best_estimator_
# score_svc = cross_val_score(model_svc, X, y)
# avg_score_svc = score_svc.mean()
# print(avg_score_svc)
# score_svcBis = cross_val_score(model_svc, X_standardR, y)
# avg_score_svcBis = score_svcBis.mean()
# print(avg_score_svcBis)
#
grid_svc.fit(X_minMax, y);
print(grid_svc.best_params_)
model_svc = grid_svc.best_estimator_
score_svc = cross_val_score(model_svc, X, y)
avg_score_svc = score_svc.mean()
print(avg_score_svc)
score_svcBis = cross_val_score(model_svc, X_minMax, y)
avg_score_svcBis = score_svcBis.mean()
print(avg_score_svcBis)
# Predict on the test and validation data.
y_test = grid_svc.predict(X_test_minMax)
y_valid = grid_svc.predict(X_valid_minMax)
# Save results
np.savetxt("protein_test_svcMinMax.predict", y_test, fmt="%d")
np.savetxt("protein_valid_svcMinMax.predict", y_valid, fmt="%d")

zip_obj = ZipFile('submission_svcMinMax.zip', 'w')
zip_obj.write("protein_test_svcMinMax.predict")
zip_obj.write("protein_valid_svcMinMax.predict")

zip_obj.close()
#
grid_svc.fit(X_maxAbs, y);
print(grid_svc.best_params_)
model_svc = grid_svc.best_estimator_
score_svc = cross_val_score(model_svc, X, y)
avg_score_svc = score_svc.mean()
print(avg_score_svc)
score_svcBis = cross_val_score(model_svc, X_maxAbs, y)
avg_score_svcBis = score_svcBis.mean()
print(avg_score_svcBis)
# Predict on the test and validation data.
y_test = grid_svc.predict(X_test_maxAbs)
y_valid = grid_svc.predict(X_valid_maxAbs)
# Save results
np.savetxt("protein_test_svcMaxAbs.predict", y_test, fmt="%d")
np.savetxt("protein_valid_svcMaxAbs.predict", y_valid, fmt="%d")

zip_obj = ZipFile('submission_svcMaxAbs.zip', 'w')
zip_obj.write("protein_test_svcMaxAbs.predict")
zip_obj.write("protein_valid_svcMaxAbs.predict")

zip_obj.close()
#
grid_svc.fit(X_robust, y);
print(grid_svc.best_params_)
model_svc = grid_svc.best_estimator_
score_svc = cross_val_score(model_svc, X, y)
avg_score_svc = score_svc.mean()
print(avg_score_svc)
score_svcBis = cross_val_score(model_svc, X_robust, y)
avg_score_svcBis = score_svcBis.mean()
print(avg_score_svcBis)
# Predict on the test and validation data.
y_test = grid_svc.predict(X_test_robust)
y_valid = grid_svc.predict(X_valid_robust)
# Save results
np.savetxt("protein_test_svcRobust.predict", y_test, fmt="%d")
np.savetxt("protein_valid_svcRobust.predict", y_valid, fmt="%d")

zip_obj = ZipFile('submission_svcRobust.zip', 'w')
zip_obj.write("protein_test_svcRobust.predict")
zip_obj.write("protein_valid_svcRobust.predict")

zip_obj.close()
#
grid_svc.fit(X_powerYJ, y);
print(grid_svc.best_params_)
model_svc = grid_svc.best_estimator_
score_svc = cross_val_score(model_svc, X, y)
avg_score_svc = score_svc.mean()
print(avg_score_svc)
score_svcBis = cross_val_score(model_svc, X_powerYJ, y)
avg_score_svcBis = score_svcBis.mean()
print(avg_score_svcBis)
# Predict on the test and validation data.
y_test = grid_svc.predict(X_test_powerYJ)
y_valid = grid_svc.predict(X_valid_powerYJ)
# Save results
np.savetxt("protein_test_svcPowerYJ.predict", y_test, fmt="%d")
np.savetxt("protein_valid_svcPowerYJ.predict", y_valid, fmt="%d")

zip_obj = ZipFile('submission_svcPowerYJ.zip', 'w')
zip_obj.write("protein_test_svcPowerYJ.predict")
zip_obj.write("protein_valid_svcPowerYJ.predict")

zip_obj.close()
#
# grid_svc.fit(X_powerBC, y);
# print(grid_svc.best_params_)
# model_svc = grid_svc.best_estimator_
# score_svc = cross_val_score(model_svc, X, y)
# avg_score_svc = score_svc.mean()
# print(avg_score_svc)
# score_svcBis = cross_val_score(model_svc, X_powerBC, y)
# avg_score_svcBis = score_svcBis.mean()
# print(avg_score_svcBis)
#
grid_svc.fit(X_quantileTU, y);
print(grid_svc.best_params_)
model_svc = grid_svc.best_estimator_
score_svc = cross_val_score(model_svc, X, y)
avg_score_svc = score_svc.mean()
print(avg_score_svc)
score_svcBis = cross_val_score(model_svc, X_quantileTU, y)
avg_score_svcBis = score_svcBis.mean()
print(avg_score_svcBis)
# Predict on the test and validation data.
y_test = grid_svc.predict(X_test_quantileTU)
y_valid = grid_svc.predict(X_valid_quantileTU)
# Save results
np.savetxt("protein_test_svcQuantileTU.predict", y_test, fmt="%d")
np.savetxt("protein_valid_svcQuantileTU.predict", y_valid, fmt="%d")

zip_obj = ZipFile('submission_svcQuantileTU.zip', 'w')
zip_obj.write("protein_test_svcQuantileTU.predict")
zip_obj.write("protein_valid_svcQuantileTU.predict")

zip_obj.close()
#
grid_svc.fit(X_quantileTN, y);
print(grid_svc.best_params_)
model_svc = grid_svc.best_estimator_
score_svc = cross_val_score(model_svc, X, y)
avg_score_svc = score_svc.mean()
print(avg_score_svc)
score_svcBis = cross_val_score(model_svc, X_quantileTN, y)
avg_score_svcBis = score_svcBis.mean()
print(avg_score_svcBis)
# Predict on the test and validation data.
y_test = grid_svc.predict(X_test_quantileTN)
y_valid = grid_svc.predict(X_valid_quantileTN)
# Save results
np.savetxt("protein_test_svcQuantileTN.predict", y_test, fmt="%d")
np.savetxt("protein_valid_svcQuantileTN.predict", y_valid, fmt="%d")

zip_obj = ZipFile('submission_svcQuantileTN.zip', 'w')
zip_obj.write("protein_test_svcQuantileTN.predict")
zip_obj.write("protein_valid_svcQuantileTN.predict")

zip_obj.close()
#
grid_svc.fit(X_normalN, y);
print(grid_svc.best_params_)
model_svc = grid_svc.best_estimator_
score_svc = cross_val_score(model_svc, X, y)
avg_score_svc = score_svc.mean()
print(avg_score_svc)
score_svcBis = cross_val_score(model_svc, X_normalN, y)
avg_score_svcBis = score_svcBis.mean()
print(avg_score_svcBis)
# Predict on the test and validation data.
y_test = grid_svc.predict(X_test_normalN)
y_valid = grid_svc.predict(X_valid_normalN)
# Save results
np.savetxt("protein_test_svcNormalN.predict", y_test, fmt="%d")
np.savetxt("protein_valid_svcNormalN.predict", y_valid, fmt="%d")

zip_obj = ZipFile('submission_svcNormalN.zip', 'w')
zip_obj.write("protein_test_svcNormalN.predict")
zip_obj.write("protein_valid_svcNormalN.predict")

zip_obj.close()
# #
# grid_svc.fit(X_normalR, y);
# print(grid_svc.best_params_)
# model_svc = grid_svc.best_estimator_
# score_svc = cross_val_score(model_svc, X, y)
# avg_score_svc = score_svc.mean()
# print(avg_score_svc)
# score_svcBis = cross_val_score(model_svc, X_normalR, y)
# avg_score_svcBis = score_svcBis.mean()
# print(avg_score_svcBis)


# # Predict on the test and validation data.
# y_test = grid_svc.predict(X_test)
# y_valid = grid_svc.predict(X_valid)
# # Save results
# np.savetxt("protein_test_svc10.predict", y_test, fmt="%d")
# np.savetxt("protein_valid_svc10.predict", y_valid, fmt="%d")
#
# zip_obj = ZipFile('submission_svc10.zip', 'w')
# zip_obj.write("protein_test_svc10.predict")
# zip_obj.write("protein_valid_svc10.predict")
#
# zip_obj.close()
