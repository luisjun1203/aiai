# 14_3 카피
from keras.models import Sequential, Model, load_model
from keras.layers import Dense, Dropout, Input, Flatten, Conv1D
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from sklearn.model_selection import StratifiedKFold, cross_val_predict, RandomizedSearchCV
from sklearn.datasets import load_diabetes
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score    # r2 결정계수
import numpy as np
import time
from sklearn.utils import all_estimators
import warnings
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import  HalvingGridSearchCV
from sklearn.pipeline import Pipeline

warnings.filterwarnings('ignore')


#1. 데이터
datasets = load_diabetes()
x = datasets.data
y = datasets.target
x_train, x_test, y_train, y_test = train_test_split(x, y,
            train_size = 0.7,
            test_size = 0.3,
            shuffle = True,
            random_state = 4567)


from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler
from sklearn.preprocessing import StandardScaler, RobustScaler


# scaler = RobustScaler()
# scaler.fit(x_train)
# x_train = scaler.transform(x_train)
# x_test = scaler.transform(x_test)
# print(np.min(x_train))  # 
# print(np.min(x_test))   # 
# print(np.max(x_train))  # 
# print(np.max(x_test))   # 


print(x)
print(y)
print(x.shape, y.shape)     # (442, 10) (442,)
print(datasets.feature_names)   # ['age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6']
print(datasets.DESCR)

x = x.reshape(-1, 10, 1)


from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold

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
                 ('RF', RandomForestClassifier())])

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
# min_resources_: 40
# max_resources_: 309
# aggressive_elimination: False
# factor: 2
# ----------
# iter: 0
# n_candidates: 76
# n_resources: 40
# Fitting 5 folds for each of 76 candidates, totalling 380 fits
# ----------
# iter: 1
# n_candidates: 38
# n_resources: 80
# Fitting 5 folds for each of 38 candidates, totalling 190 fits
# ----------
# iter: 2
# n_candidates: 19
# n_resources: 160
# Fitting 5 folds for each of 19 candidates, totalling 95 fits
# 최적의 매개변수 :  RandomForestClassifier(max_depth=8, min_samples_leaf=5)
# 최적의 파라미터 :  {'max_depth': 8, 'min_samples_leaf': 5}
# best_score :  0.01875
# model_score :  0.015037593984962405
# accuracy_score 0.015037593984962405
# 최적 튠 ACC :  0.015037593984962405
# 걸린신간 :  14.32 초
# PS C:\Study>






