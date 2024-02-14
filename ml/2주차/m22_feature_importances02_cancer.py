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


#1. 데이터
datasets= load_breast_cancer()

X = datasets.data
y = datasets.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, shuffle=False, random_state=3)

mms = MinMaxScaler()
mms.fit(X_train)
X_train = mms.transform(X_train)
X_test = mms.transform(X_test)

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




# MinMaxScaler
# 정확도 :  0.9956140350877193
# 로스 :  [0.016668006777763367, 0.9956140518188477]
# R2 :  0.9760378349973726

# MaxAbsScaler
# 정확도 :  0.956140350877193
# 로스 :  [0.5653559565544128, 0.9561403393745422]
# R2 :  0.7603783499737258

# StandardScaler
# 정확도 :  0.9780701754385965
# 로스 :  [0.303570419549942, 0.9780701994895935]
# R2 :  0.8801891749868629

# # RobustScaler
# 정확도 :  0.9692982456140351
# 로스 :  [0.09083357453346252, 0.969298243522644]
# R2 :  0.832264844981608

# linearsvc
# 정확도 :  0.9649122807017544
# model.score :  0.9649122807017544
# R2 :  0.8083026799789806


# RandomForestClassifier accuracy: 0.9605
# RandomForestClassifier : [0.04591717 0.01677276 0.04519297 0.03166873 0.00933174 0.00899141
#  0.03902775 0.11763835 0.00259388 0.00397243 0.00785815 0.00487554
#  0.01326458 0.01993654 0.00575058 0.00491898 0.00545244 0.00329444
#  0.00440461 0.00510406 0.13853722 0.01488148 0.11591062 0.13169502
#  0.0149445  0.02117601 0.04719356 0.10673662 0.00733559 0.00562228]
