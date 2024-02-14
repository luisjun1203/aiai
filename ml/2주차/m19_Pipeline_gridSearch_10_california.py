# 14_2 카피
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input, Conv1D, Flatten, SimpleRNN
from sklearn.model_selection import train_test_split, KFold, cross_val_score, cross_val_predict,GridSearchCV, RandomizedSearchCV
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import r2_score, mean_squared_error,accuracy_score
import numpy as np
import time
from sklearn.utils import all_estimators
import warnings
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import  HalvingGridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
warnings.filterwarnings('ignore')



#1. 데이터
datasets = fetch_california_housing()
x = datasets.data
y = datasets.target
print(x.shape, y.shape) # (20640, 8) (20640,)
print(datasets.feature_names)
# ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'Latitude', 'Longitude']
print(datasets.DESCR)   # 20640 행 , 8 열

# [실습] 만들기
# R2 0.55 ~ 0.6 이상

x_train, x_test, y_train, y_test = train_test_split(x, y,
            train_size = 0.7,
            test_size = 0.3,
            shuffle = True,
            random_state = 4567)

print(x_train.shape, y_train.shape) # (14447, 8) (14447,)
# x_train = x_train.reshape(-1,8,1)
# x_test = x_test.reshape(-1,8,1)

n_splits = 5
# kfold = KFold(n_splits=n_splits, shuffle=True, random_state=123)
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=123)


from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

parameters = [
    {'RF__n_estimators':[100,200], 'RF__max_depth':[6,10,12], 'RF__min_samples_leaf':[3,10]},
    {'RF__max_depth':[6,8,10,12], 'RF__min_samples_leaf':[3,5,7,10]},
    {'RF__min_samples_leaf':[3,5,7,10], 'RF__min_samples_split':[2,3,5,10]},
    {'RF__min_samples_leaf':[3,5,7,10], 'RF__min_samples_split':[2,3,5,10]},
    {'RF__min_samples_split':[2,3,5,10]},
    {'RF__min_samples_split':[2,3,5,10]}
]


# 2 모델
# model = SVC(C=1, kernel='linear', degree=3)
print('==============하빙그리드서치 시작==========================')
pipe = Pipeline([('MM', MinMaxScaler()),
                 ('RF', RandomForestRegressor())])

model = HalvingGridSearchCV(pipe, parameters,
                     cv = kfold,
                     verbose=1,
                    #  refit=True # 디폴트 트루 # 한바퀴 돌린후 다시 돌린다
                     n_jobs=3   # 24개의 코어중 3개 사용 / 전부사용 -1
                     , random_state= 66,
                    # n_iter=10 # 디폴트 10
                    factor=2,
                    min_resources=40)


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
# print('accuracy_score', accuracy_score(y_test, y_predict))

y_pred_best = model.best_estimator_.predict(x_test)
            # SVC(C-1, kernel='linear').predicict(x_test)
# print('최적 튠 ACC : ', accuracy_score(y_test, y_pred_best))

print('걸린신간 : ', round(end_time - start_time, 2), '초')

# import pandas as pd
# print(pd.DataFrame(model.cv_results_).T)


# ==============하빙그리드서치 시작==========================
# n_iterations: 3
# n_required_iterations: 4
# n_possible_iterations: 3
# min_resources_: 1000
# max_resources_: 14447
# aggressive_elimination: False
# factor: 3
# ----------
# iter: 0
# n_candidates: 76
# n_resources: 1000
# Fitting 5 folds for each of 76 candidates, totalling 380 fits
# ----------
# iter: 1
# n_candidates: 26
# n_resources: 3000
# Fitting 5 folds for each of 26 candidates, totalling 130 fits
# ----------
# iter: 2
# n_candidates: 9
# n_resources: 9000
# Fitting 5 folds for each of 9 candidates, totalling 45 fits
# 최적의 매개변수 :  RandomForestRegressor(min_samples_leaf=3, min_samples_split=3)
# 최적의 파라미터 :  {'min_samples_leaf': 3, 'min_samples_split': 3}
# best_score :  0.7871390035304324
# model_score :  0.8000304821822726
# 걸린신간 :  71.63 초
# PS C:\Study> 