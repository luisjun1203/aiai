from sklearn.datasets import load_diabetes
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import time
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, StandardScaler, RobustScaler
from sklearn.metrics import r2_score, mean_squared_error,mean_squared_log_error,accuracy_score


datasets = load_diabetes()
X = datasets.data
y = datasets.target


# columns = datasets.feature_names
# X = pd.DataFrame(X,columns=columns)


# fi_str = "0.11132886 0.02855667 0.12190132 0.11529632 0.11109842 0.11213265\
#  0.10702958 0.05943837 0.12141634 0.11180148"


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


# lae = LabelEncoder()
# lae.fit(y)
# y = lae.transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=1226)           # 1226 713

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
    r2 = r2_score(y_test, y_predict)
    
   
    accuracy_results[n] = r2
    print(f"n_components = {n}, Accuracy: {r2}")

# loss = 'mse' , random_state=1226, epochs=50, batch_size=10, test_size=0.1
# 로스 :  2031.322509765625+
# R2스코어 :  0.7246106597957658

# model.score :  0.022222222222222223
# R2스코어 :  0.5699312314693923

# RandomForestClassifier accuracy: 0.4862
# RandomForestClassifier : [0.11132886 0.02855667 0.12190132 0.11529632 0.11109842 0.11213265
#  0.10702958 0.05943837 0.12141634 0.11180148]


# RandomForestClassifier accuracy: 0.1617
# RandomForestClassifier : [0.12854363 0.1383427  0.12700509 0.12773477 0.13306956 0.08066242
#  0.13709115 0.12755068]

# n_components = 1, Accuracy: -0.07367497259498124
# n_components = 2, Accuracy: -0.20828845415850794
# n_components = 3, Accuracy: 0.16972367781352338
# n_components = 4, Accuracy: 0.07465551095333045
# n_components = 5, Accuracy: 0.06914225180718536
# n_components = 6, Accuracy: 0.05729883188976148
# n_components = 7, Accuracy: 0.19606462110654854
# n_components = 8, Accuracy: 0.16249062577235995
# n_components = 9, Accuracy: 0.16259149824024088
# n_components = 10, Accuracy: 0.033387096060949695