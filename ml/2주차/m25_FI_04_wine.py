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


models = [
DecisionTreeClassifier(),
RandomForestClassifier(),
GradientBoostingClassifier(),
XGBClassifier()
]


for model in models:
    model_name = model.__class__.__name__
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f"{model_name} accuracy: {accuracy:.4f}")
    print(model.__class__.__name__, ":", model.feature_importances_)


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

























