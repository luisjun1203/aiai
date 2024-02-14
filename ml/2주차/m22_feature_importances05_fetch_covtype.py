from sklearn.datasets import fetch_covtype
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


datasets = fetch_covtype()

X = datasets.data
y = datasets.target


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35, shuffle=True, random_state=3, stratify=y)

# print(X_train.shape, X_test.shape)      # (464809, 54) (116203, 54)
# print(y_train.shape, y_test.shape)      # (464809, 7) (116203, 7)


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



# 로스 :  0.6938604116439819
# ACC :  0.701832115650177
# accuracy_score :  0.7018321385850623

# 로스 :  0.5239526033401489
# ACC :  0.7768474221229553
# accuracy_score :  0.7768474135779627



# model.score :  0.5275601780138182
# accuracy_score :  0.5275601780138182



# RandomForestClassifier accuracy: 0.9508
# RandomForestClassifier : [2.35706587e-01 4.81478779e-02 3.32470063e-02 6.10893116e-02
#  5.77017224e-02 1.16636661e-01 4.16720995e-02 4.39262560e-02
#  4.21169150e-02 1.09083355e-01 1.01819864e-02 6.03515862e-03
#  1.17111712e-02 3.68831587e-02 1.04801852e-03 9.03287542e-03
#  2.39440367e-03 1.30093885e-02 5.21761227e-04 2.65013408e-03
#  9.33725677e-06 4.38580427e-05 1.50606890e-04 1.09361901e-02
#  2.81691827e-03 1.06806535e-02 4.08679953e-03 3.32608552e-04
#  2.63036835e-06 8.15618286e-04 1.70397697e-03 2.27480557e-04
#  9.94458692e-04 1.92167430e-03 7.12940272e-04 1.45258061e-02
#  1.03483542e-02 4.11028817e-03 1.45009886e-04 4.43341498e-04
#  6.81962105e-04 1.93598042e-04 5.53663171e-03 3.16130349e-03
#  3.66774796e-03 5.55543689e-03 4.67047734e-03 6.06793266e-04
#  1.55982621e-03 8.50360573e-05 6.42950975e-04 9.86635877e-03
#  1.00973944e-02 5.87008309e-03]




