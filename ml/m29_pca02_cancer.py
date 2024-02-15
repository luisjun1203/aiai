import numpy as np
from sklearn.datasets import load_breast_cancer
from keras.models import Sequential
from keras.layers import Dense
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error,mean_squared_log_error,accuracy_score
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, StandardScaler, RobustScaler
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.decomposition import PCA

#1. 데이터
datasets= load_breast_cancer()

X = datasets.data
y = datasets.target

# columns = datasets.feature_names
# X = pd.DataFrame(X,columns=columns)

# print(datasets.feature_names)
# ['mean radius' 'mean texture' 'mean perimeter' 'mean area'
#  'mean smoothness' 'mean compactness' 'mean concavity'
#  'mean concave points' 'mean symmetry' 'mean fractal dimension'
#  'radius error' 'texture error' 'perimeter error' 'area error'
#  'smoothness error' 'compactness error' 'concavity error'
#  'concave points error' 'symmetry error' 'fractal dimension error'
#  'worst radius' 'worst texture' 'worst perimeter' 'worst area'
#  'worst smoothness' 'worst compactness' 'worst concavity'
#  'worst concave points' 'worst symmetry' 'worst fractal dimension']

# RandomForestClassifier : [0.04591717 0.01677276 0.04519297 0.03166873 0.00933174 0.00899141
#  0.03902775 0.11763835 0.00259388 0.00397243 0.00785815 0.00487554
#  0.01326458 0.01993654 0.00575058 0.00491898 0.00545244 0.00329444
#  0.00440461 0.00510406 0.13853722 0.01488148 0.11591062 0.13169502
#  0.0149445  0.02117601 0.04719356 0.10673662 0.00733559 0.00562228]

# X = np.delete(X, 0, axis=1)
# columns = 

# X = pd.DataFrame(X)

# X = X.drop(['mean radius','mean texture'], axis=1)

# fi_str = "0.04591717 0.01677276 0.04519297 0.03166873 0.00933174 0.00899141\
#  0.03902775 0.11763835 0.00259388 0.00397243 0.00785815 0.00487554\
#  0.01326458 0.01993654 0.00575058 0.00491898 0.00545244 0.00329444\
#  0.00440461 0.00510406 0.13853722 0.01488148 0.11591062 0.13169502\
#  0.0149445  0.02117601 0.04719356 0.10673662 0.00733559 0.00562228"


# fi_str = fi_str.split()
# fi_float = [float(s) for s in fi_str]
# print(fi_float)
# fi_list = pd.Series(fi_float)

# low_idx_list = fi_list[fi_list <= fi_list.quantile(0.25)].index
# print('low_idx_list',low_idx_list)

# low_col_list = [X.columns[index] for index in low_idx_list]
# if len(low_col_list) > len(X.columns) * 0.25:
#     low_col_list = low_col_list[:int(len(X.columns)*0.25)]
# print('low_col_list',low_col_list)
# X.drop(low_col_list,axis=1,inplace=True)
# print("after X.shape",X.shape)






X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, shuffle=False, random_state=3)

sts = StandardScaler()
sts.fit(X_train)
X_train = sts.transform(X_train)
X_test = sts.transform(X_test)


n_features = X_train.shape[1]
accuracy_results = {}

for n in range(1, n_features + 1):
    pca = PCA(n_components=n)
    X_train_p = pca.fit_transform(X_train)
    X_test_p = pca.transform(X_test)  

    model = RandomForestClassifier(random_state=3)
    model.fit(X_train_p, y_train)
    y_predict = model.predict(X_test_p)
    acc = accuracy_score(y_test, y_predict)
    
   
    accuracy_results[n] = acc
    print(f"n_components = {n}, Accuracy: {acc}")
    
    
# n_components = 1, Accuracy: 0.9035087719298246
# n_components = 2, Accuracy: 0.9429824561403509
# n_components = 3, Accuracy: 0.9429824561403509
# n_components = 4, Accuracy: 0.9473684210526315
# n_components = 5, Accuracy: 0.9429824561403509
# n_components = 6, Accuracy: 0.9385964912280702
# n_components = 7, Accuracy: 0.9385964912280702
# n_components = 8, Accuracy: 0.9473684210526315
# n_components = 9, Accuracy: 0.9429824561403509
# n_components = 10, Accuracy: 0.9429824561403509
# n_components = 11, Accuracy: 0.9473684210526315
# n_components = 12, Accuracy: 0.9342105263157895
# n_components = 13, Accuracy: 0.9298245614035088
# n_components = 14, Accuracy: 0.9473684210526315
# n_components = 15, Accuracy: 0.9517543859649122
# n_components = 16, Accuracy: 0.9429824561403509
# n_components = 17, Accuracy: 0.9429824561403509
# n_components = 18, Accuracy: 0.9429824561403509
# n_components = 19, Accuracy: 0.9298245614035088
# n_components = 20, Accuracy: 0.9298245614035088
# n_components = 21, Accuracy: 0.9342105263157895
# n_components = 22, Accuracy: 0.9385964912280702
# n_components = 23, Accuracy: 0.9342105263157895
# n_components = 24, Accuracy: 0.9298245614035088
# n_components = 25, Accuracy: 0.9210526315789473
# n_components = 26, Accuracy: 0.9254385964912281
# n_components = 27, Accuracy: 0.9254385964912281
# n_components = 28, Accuracy: 0.9429824561403509
# n_components = 29, Accuracy: 0.9342105263157895
# n_components = 30, Accuracy: 0.9122807017543859