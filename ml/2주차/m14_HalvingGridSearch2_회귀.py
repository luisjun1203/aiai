import numpy as np
from sklearn.datasets import load_iris, load_digits, load_diabetes
from sklearn.svm import SVC, SVR
from sklearn.model_selection import train_test_split, KFold, cross_val_score, StratifiedKFold, cross_val_predict
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, r2_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV
import time

#1. 데이터
# x, y = load_iris(return_X_y=True)
# x, y = load_digits(return_X_y=True)
x, y = load_diabetes(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(x,y, random_state=123, train_size=0.8)

print(x_train.shape)    #(1437, 64)

scaler =  StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

# n_splits = 5
# kfold = KFold(n_splits=n_splits, shuffle=True, random_state=123)
    
parameters = [
    {"C":[1, 10, 100, 1000], "kernel":["linear"], "degree":[3,4,5]}
    , {"C":[1, 10, 100], "kernel":["rbf"], "gamma":[0.001, 0.0001]}
    , {"C":[1, 10, 100, 1000], "kernel":["sigmoid"], "gamma":[0.01, 0.001, 0.0001], "degree":[3, 4]}
         
]
   
print("===============================하빙 그리드 시작=====================")
model = HalvingGridSearchCV(SVR()
                     , parameters
                     , cv=5
                     , verbose=1
                     , refit=True   
                     , n_jobs=-1  
                     , random_state=66
                     , factor=3 # 디폴트 3
                    #  , min_resources=150
                    #  , n_iter=20    # 디폴트 10
                     )

start_time = time.time()
model.fit(x_train, y_train)
end_time = time.time()
#3
print(f"최적의 매개변수 : {model.best_estimator_}\n최적의 파라미터 : {model.best_params_}")
print(f"best score : {model.best_score_}\nmodel.score : {model.score(x_test, y_test)}")




y_predict = model.predict(x_test)


y_predict_best = model.best_estimator_.predict(x_test)

print(f"걸린 시간 : {round(end_time - start_time, 2)} 초")

import pandas as pd
# print(pd.DataFrame(model.cv_results_).T)

import sklearn as sk
print(sk.__version__)   #1.1.3