# 1. factor 조절
# 2. min_resources 조절
# # 훈련데이터를 iter_2 이상 돌려라

import numpy as np
from sklearn.datasets import load_iris
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, KFold, cross_val_score, cross_val_predict
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import  HalvingGridSearchCV
from sklearn.pipeline import make_pipeline

import time

# 1 데이터
x, y = load_iris(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, random_state=123, train_size=0.8, stratify=y
    )

n_splits = 5
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=123)

# parameters = [
#     {'n_estimators':[100,200], 'max_depth':[6,10,12], 'min_samples_leaf':[3,10]},
#     {'max_depth':[6,8,10,12], 'min_samples_leaf':[3,5,7,10]},
#     {'min_samples_leaf':[3,5,7,10], 'min_samples_split':[2,3,5,10]},
#     {'min_samples_leaf':[3,5,7,10], 'min_samples_split':[2,3,5,10]},
#     {'min_samples_split':[2,3,5,10]},
#     {'n_jobs':[-1,2,4],'min_samples_split':[2,3,5,10]}
# ]

parameters = [
    {'n_estimators':[100,200], 'max_depth':[6,10,12], 'min_samples_leaf':[3,10]},
    {'max_depth':[6,8,10,12], 'min_samples_leaf':[3,5,7,10]},
    {'min_samples_leaf':[3,5,7,10], 'min_samples_split':[2,3,5,10]},
    {'min_samples_leaf':[3,5,7,10], 'min_samples_split':[2,3,5,10]},
    {'min_samples_split':[2,3,5,10]},
    {'n_jobs':[-1],'min_samples_split':[2,3,5,10]}
]

# 2 모델
# model = SVC(C=1, kernel='linear', degree=3)
print('==============하빙그리드서치 시작==========================')
model = make_pipeline(MinMaxScaler(), HalvingGridSearchCV(RandomForestClassifier(), 
                     parameters,
                     cv = kfold,
                     verbose=1,
                    #  refit=True # 디폴트 트루 # 한바퀴 돌린후 다시 돌린다
                     n_jobs=3,   # 24개의 코어중 3개 사용 / 전부사용 -1
                     random_state=66,
                     factor=2
                    #  min_resources=30
                     
                     ))


start_time = time.time()
model.fit(x_train, y_train)
end_time = time.time()

print('최적의 매개변수 : ', model.best_estimator_)
# 최적의 매개변수 :  SVC(C=1, kernel='linear')
print('최적의 파라미터 : ', model.best_params_) # 내가 선택한것
# 최적의 파라미터 :  {'C': 1, 'degree': 3, 'kernel': 'linear'} 우리가 지정한거중에 가장 좋은거
print('best_score : ', model.best_score_)   # 핏한거의 최고의 스코어
# best_score :  0.975
print('model_score : ', model.score(x_test, y_test))    # 
# model_score :  0.9666666666666667


y_predict = model.predict(x_test)
print('accuracy_score', accuracy_score(y_test, y_predict))

y_pred_best = model.best_estimator_.predict(x_test)
            # SVC(C-1, kernel='linear').predicict(x_test)
print('최적 튠 ACC : ', accuracy_score(y_test, y_pred_best))

print('걸린신간 : ', round(end_time - start_time, 2), '초')

# import pandas as pd
# print(pd.DataFrame(model.cv_results_).T)



# ==============하빙그리드서치 시작==========================
# n_iterations: 3
# n_required_iterations: 7
# n_possible_iterations: 3
# min_resources_: 30
# max_resources_: 120
# aggressive_elimination: False
# factor: 2
# ----------
# iter: 0
# n_candidates: 68
# n_resources: 30
# Fitting 5 folds for each of 68 candidates, totalling 340 fits
# ----------
# iter: 1
# n_candidates: 34
# n_resources: 60
# Fitting 5 folds for each of 34 candidates, totalling 170 fits
# ----------
# iter: 2
# n_candidates: 17
# n_resources: 120
# Fitting 5 folds for each of 17 candidates, totalling 85 fits
# 최적의 매개변수 :  RandomForestClassifier(min_samples_leaf=5)
# 최적의 파라미터 :  {'min_samples_leaf': 5, 'min_samples_split': 2}
# best_score :  0.9583333333333334
# model_score :  0.9333333333333333
# accuracy_score 0.9333333333333333
# 최적 튠 ACC :  0.9333333333333333
# 걸린신간 :  11.39 초
# PS C:\Study> 