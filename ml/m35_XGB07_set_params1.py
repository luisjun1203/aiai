from sklearn.datasets import load_diabetes
from xgboost import XGBClassifier, XGBRegressor
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV, HalvingRandomSearchCV, GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import accuracy_score, r2_score
import time

#1.데이터
X,y = load_diabetes(return_X_y=True)

print(X.shape)  

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=3)

n_splits= 5
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=123)

mms = MinMaxScaler()
mms.fit(X_train)
X_train = mms.transform(X_train)
X_test = mms.transform(X_test)

########################## XGB 하이퍼 파라미터 ##############################################

#    'n_estimators': [100, 200, 300, 400, 500, 1000],  # 부스팅 라운드의 수 / 디폴트 100 / 1 ~ inf/ 정수
#     'learning_rate': [0.1, 0.2, 0.3, 0.5, 1, 0.01, 0.05, 0.001],  # 학습률 / 디폴트 0.3 / 0 ~ 1 / eta / 통상적으로 작으면 작을수록 좋다
#     'max_depth': [None, 2, 3, 4, 5, 6, 7, 8, 9, 10],  # 트리의 최대 깊이 / 디폴트 6 / 0 ~ inf/ 정수
#     'min_child_weight': [0, 0.01, 0.001, 0.1, 0.5, 1, 5, 10, 100],  # 자식에 필요한 모든 관측치에 대한 가중치 합의 최소/ 디폴트 1 / 0~inf
#     'gamma': [0, 0.5, 1, 1.5, 2, 10, 100],  # 리프 노드를 추가적으로 나눌지 결정하기 위한 최소 손실 감소/ 디폴트 0 / 0~ inf
#     'subsample': [0.6, 0.8, 1.0],  # 각 트리마다의 관측 데이터 샘플링 비율 / 디폴트 1 / 0 ~ 1
#     'colsample_bytree': [0, 0.6, 0.8, 1.0],  # 각 트리 구성에 필요한 컬럼(특성) 샘플링 비율 / 디폴트 1 / 0 ~ 1
#     'colsample_bylevel': [0, 0.6, 0.8, 1.0], #  디폴트 1 / 0 ~ 1
#     'colsample_bynode': [0, 0.6, 0.8, 1.0], #  디폴트 1 / 0 ~ 1
#     'reg_alpha' : [0, 0.1, 0.01, 0.001, 1, 2, 10],   # 디폴트 0 / 0 ~ inf / L1 절대값 가중치 규제 / alpha
#     'reg_lambda' :   [0, 0.1, 0.01, 0.001, 1, 2, 10],   # 디폴트 1 / 0 ~ inf / L2 제곱 가중치 규제 / lambda
#     'objective': ['multi:softmax'],  # 학습 태스크 파라미터
#     'num_class': [30],
#     'verbosity' : [1] 





parameters = {
    'n_estimators': [10, 30, 50, 100, 200, 300, 500, 1000],  # 부스팅 라운드의 수/ 디폴트 100/ 1 ~ inf/ 정수
    'learning_rate': [0.001, 0.01, 0.05, 0.1, 0.5, 1],  # 학습률/ 디폴트 0.3/0~1/
    'max_depth': [3],  # 트리의 최대 깊이/ 디폴트 6/ 0 ~ inf/ 정수
    'min_child_weight': [5],  # 자식에 필요한 모든 관측치에 대한 가중치 합의 최소/ 디폴트 1 / 0~inf
    'gamma': [1],  # 리프 노드를 추가적으로 나눌지 결정하기 위한 최소 손실 감소/ 디폴트 0/ 0~ inf
    'subsample': [0.6],  # 각 트리마다의 관측 데이터 샘플링 비율/ 디폴트 1 / 0~1
    'colsample_bytree': [0.6],  # 각 트리 구성에 필요한 컬럼(특성) 샘플링 비율/ 디폴트 1 / 0~1
    'colsample_bylevel': [0.6], #  디폴트 1 / 0~1
    'colsample_bynode': [0.6], #  디폴트 1 / 0~1
    'reg_alpha' : [0],   # 디폴트 0 / 0 ~ inf / L1 절대값 가중치 규제(제한) / alpha
    'reg_lambda' :   [1],   # 디폴트 1 / 0 ~ inf / L2 제곱 가중치 규제(제한) / lambda
    'objective': ['multi:softmax'],  # 학습 태스크 파라미터
    'num_class': [30],
    'verbosity' : [1] 
}

#2. 모델 구성
# model = GridSearchCV(XGBClassifier(), parameters, cv=kfold, verbose=1,
                    # refit = True,     # default
                    #  n_jobs=22)

model = XGBRegressor()

# 3. 훈련
start = time.time()

model.fit(X_train, y_train)

# 4. 평가, 예측
# print("최적의 매개변수 : ", model.best_estimator_)              #

# print("최적의 파라미터 : ", model.best_params_)                 #

# print('best_score : ', model.best_score_)
results = model.score(X_test, y_test)
print("최종점수1 : ", results)
# y_predict = model.predict(X_test)
# r2 = r2_score(y_test, y_predict)
# print("r2_score : ", r2)

# y_pred_best = model.best_estimator_.predict(X_test)
# print("최적튠 R2 : " , r2_score(y_test, y_pred_best))
# end = time.time()
# print("걸린시간 : ", round(end- start, 2), "초")


model.set_params(gamma=0.3)


model.fit(X_train, y_train)


results = model.score(X_test, y_test)
print("최종점수2 : ", results)
# y_predict = model.predict(X_test)
# r2 = r2_score(y_test, y_predict)
# print("r2_score : ", r2)

model.set_params(learning_rate = 0.01)


model.fit(X_train, y_train)


results = model.score(X_test, y_test)
print("최종점수3 : ", results)

model.set_params(learning_rate = 0.005, n_estimators = 400,
                 max_depth = 7, gamma = 0.1,reg_alpha = 0.5 ,
                 reg_lambda = 0.7, min_child_weight=1,
                 
                 
                 
                 )

model.fit(X_train, y_train)


results = model.score(X_test, y_test)
print("최종점수4 : ", results)
print("사용된 파라미터 : ", model.get_params()) #모델에 사용된 파라미터들 알아보기

# model.set_params(n_estimators = 300)

# model.fit(X_train, y_train)


# results = model.score(X_test, y_test)
# print("최종점수4 : ", results)





















