from sklearn.datasets import load_wine
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from keras.utils import to_categorical
from sklearn.preprocessing import OneHotEncoder
from keras. callbacks import EarlyStopping
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, StandardScaler, RobustScaler
from sklearn.metrics import r2_score, mean_squared_error,mean_squared_log_error,accuracy_score

datasets = load_wine()

X = datasets.data
y = datasets.target


columns = datasets.feature_names
X = pd.DataFrame(X,columns=columns)


fi_str = "0.12015032 0.03930748 0.01282976 0.03412675 0.02579797 0.0558285\
 0.18350481 0.01135927 0.02305502 0.1368655  0.06876278 0.09845979\
 0.18995205"


fi_str = fi_str.split()
fi_float = [float(s) for s in fi_str]
print(fi_float)
fi_list = pd.Series(fi_float)

low_idx_list = fi_list[fi_list <= fi_list.quantile(0.25)].index
print('low_idx_list',low_idx_list)

low_col_list = [X.columns[index] for index in low_idx_list]
if len(low_col_list) > len(X.columns) * 0.25:
    low_col_list = low_col_list[:int(len(X.columns)*0.25)]
print('low_col_list',low_col_list)
X.drop(low_col_list,axis=1,inplace=True)
print("after X.shape",X.shape)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, shuffle=True, random_state=713, stratify=y)


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


# 로스 :  0.2964943051338196
# ACC :  0.8888888955116272
# accuracy_score :  0.8888888888888888

# model.score :  0.8888888888888888
# accuracy_score :  0.8888888888888888



# RandomForestClassifier accuracy: 0.9630
# RandomForestClassifier : [0.12015032 0.03930748 0.01282976 0.03412675 0.02579797 0.0558285
#  0.18350481 0.01135927 0.02305502 0.1368655  0.06876278 0.09845979
#  0.18995205]

# RandomForestClassifier accuracy: 0.9259
# RandomForestClassifier : [0.13800937 0.03656422 0.02004416 0.03618777 0.17945548 0.01618698
#  0.16612505 0.07979879 0.13148506 0.19614314]
# GradientBoostingClassifier accuracy: 0.9259
# GradientBoostingClassifier : [2.73823605e-02 5.21003200e-02 1.31248938e-03 1.05899094e-04
#  2.75461808e-01 5.87010693e-04 2.93116570e-01 4.33850730e-02
#  5.49812329e-04 3.05998657e-01]
# XGBClassifier accuracy: 0.9259
# XGBClassifier : [0.05340667 0.06567135 0.         0.0162901  0.11620031 0.03275065
#  0.15221831 0.04854552 0.38795173 0.12696531]


# n_components = 1, Accuracy: 0.8148148148148148
# n_components = 2, Accuracy: 1.0
# n_components = 3, Accuracy: 1.0
# n_components = 4, Accuracy: 1.0
# n_components = 5, Accuracy: 1.0
# n_components = 6, Accuracy: 1.0
# n_components = 7, Accuracy: 1.0
# n_components = 8, Accuracy: 1.0
# n_components = 9, Accuracy: 1.0
# n_components = 10, Accuracy: 1.0

EVR = pca.explained_variance_ratio_         
print(EVR)

evr_cumsum = np.cumsum(EVR)
print(evr_cumsum)

print(sum(EVR)) 
# [0.42534439 0.65297317 0.76302249 0.8319788  0.88595072 0.92090522
#  0.94720257 0.97294205 0.98908561 1.        ]



















