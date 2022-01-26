import numpy as np
from zipfile import ZipFile
from sklearn import linear_model


from sklearn import metrics
import pandas as pd

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score

from sklearn import preprocessing
from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import PCA

from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

# Liste des pré-traitements utilisés dans le cadre du GridSearchCV:
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
scaler = preprocessing.StandardScaler() # Pour MLP
X_standard = scaler.fit_transform(X)
X_test_standard = scaler.transform(X_test)
X_valid_standard = scaler.transform(X_valid)
pca = PCA(n_components=100)
X_standardN = pca.fit_transform(X_standard)
X_test_standardN = pca.transform(X_test_standard)
X_valid_standardN = pca.transform(X_valid_standard)

scaler = preprocessing.PowerTransformer(method="yeo-johnson") # Pour KNN
X_powerYJ = scaler.fit_transform(X)
X_test_powerYJ = scaler.transform(X_test)
X_valid_powerYJ = scaler.transform(X_valid)
pca = PCA(n_components=100) # Même PCA, répétée ici pour la lisibilité unitaire du bloc
X_powerYJ = pca.fit_transform(X_powerYJ)
X_test_powerYJ = pca.transform(X_test_powerYJ)
X_valid_powerYJ = pca.transform(X_valid_powerYJ)

param_grid_knn = {'n_neighbors': [4],
              'weights':['distance'],
              'algorithm': ['ball_tree'],
              'leaf_size': [1],
              'p': [2],
              'n_jobs': [-1]}

param_grid_MLP = {'hidden_layer_sizes': [(1000,1000,1000)],
                 'activation': ['relu'],
                 #'alpha': [0.0001,0.0005,0.00005],
                 'learning_rate': ['constant'],
                 #'learning_rate_init': [0.001,0.005,0.0005],
                 'solver': ['lbfgs'],
                 'max_iter': [500]}

# Partie MLP
grid_MLP = GridSearchCV(MLPClassifier(), param_grid_MLP, cv=10, n_jobs=-1)
grid_MLP.fit(X_standardN, y);
print(grid_MLP.best_params_)
model_MLP = grid_MLP.best_estimator_
score_MLP = cross_val_score(model_MLP, X, y)
avg_score_MLP = score_MLP.mean()
print(avg_score_MLP)
score_MLPBis = cross_val_score(model_MLP, X_standardN, y)
avg_score_MLPBis = score_MLPBis.mean()
print(avg_score_MLPBis) # Le score à comparer avec l'évaluation Codalab
# Predict on the test and validation data.
y_test = grid_MLP.predict(X_test_standardN)
y_valid = grid_MLP.predict(X_valid_standardN)
# Save results
np.savetxt("protein_test_MLPStandard100Final.predict", y_test, fmt="%d")
np.savetxt("protein_valid_MLPStandard100Final.predict", y_valid, fmt="%d")

zip_obj = ZipFile('submission_MLPStandard100Final.zip', 'w')
zip_obj.write("protein_test_MLPStandard100Final.predict")
zip_obj.write("protein_valid_MLPStandard100Final.predict")

zip_obj.close()

# Partie KNN
grid_knn = GridSearchCV(KNeighborsClassifier(), param_grid_knn, cv=10, n_jobs=-1)
grid_knn.fit(X_powerYJ, y);
print(grid_knn.best_params_)
model_knn = grid_knn.best_estimator_
score_knn = cross_val_score(model_knn, X, y)
avg_score_knn = score_knn.mean()
print(avg_score_knn)
score_knnBis = cross_val_score(model_knn, X_powerYJ, y)
avg_score_knnBis = score_knnBis.mean()
print(avg_score_knnBis) # Le score à comparer avec l'évaluation Codalab
# Predict on the test and validation data.
y_test = grid_knn.predict(X_test_powerYJ)
y_valid = grid_knn.predict(X_valid_powerYJ)
# Save results
np.savetxt("protein_test_knnPowerYJ100Final.predict", y_test, fmt="%d")
np.savetxt("protein_valid_knnPowerYJ100Final.predict", y_valid, fmt="%d")

zip_obj = ZipFile('submission_knnPowerYJ100Final.zip', 'w')
zip_obj.write("protein_test_knnPowerYJ100Final.predict")
zip_obj.write("protein_valid_knnPowerYJ100Final.predict")

zip_obj.close()
