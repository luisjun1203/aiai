from sklearn.datasets import load_digits
from xgboost import XGBClassifier, XGBRFRegressor
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV, HalvingRandomSearchCV, GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import accuracy_score
import time


# 1 데이터
datasets = load_digits()

x = datasets.data
y = datasets.target

# print(x)
# print(y)    # [0 1 2 ... 8 9 8]
# print(x.shape)  # (1797, 64)    # 64니까 8 * 8
# print(y.shape)  # (1797,)
# print(pd.value_counts(y, sort=False))


x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, random_state=123, train_size=0.8, stratify=y
    )

n_splits = 5
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=123)

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

# 그리드로 돌렸을때 5 * 42 횟수.

# 2 모델
# model = RandomForestClassifier(C=1, kernel='linear', degree=3)
model = RandomizedSearchCV(XGBClassifier(), 
                     parameters,
                     cv = kfold,
                     verbose=1,
                    #  refit=True # 디폴트 트루 # 한바퀴 돌린후 다시 돌린다
                     n_jobs=3   # 24개의 코어중 3개 사용 / 전부사용 -1
                     , random_state=123, n_iter=10)

# model = RandomizedSearchCV(SVC(), 
#                      parameters,
#                      cv = kfold,
#                      verbose=1,
#                     #  refit=True # 디폴트 트루 # 한바퀴 돌린후 다시 돌린다
#                      n_jobs=3   # 24개의 코어중 3개 사용 / 전부사용 -1
#                      )




start_time = time.time()
model.fit(x_train, y_train)
end_time = time.time()

print('최적의 매개변수 : ', model.best_estimator_)

print('최적의 파라미터 : ', model.best_params_) # 내가 선택한것

print('best_score : ', model.best_score_)   # 핏한거의 최고의 스코어

print('model_score : ', model.score(x_test, y_test))    # 



y_predict = model.predict(x_test)
print('accuracy_score', accuracy_score(y_test, y_predict))

y_pred_best = model.best_estimator_.predict(x_test)
           
print('최적 튠 ACC : ', accuracy_score(y_test, y_pred_best))

print('걸린신간 : ', round(end_time - start_time, 2), '초')

# 최적의 파라미터 :  {'verbosity': 1, 'subsample': 0.6, 'reg_lambda': 1, 'reg_alpha': 0, 'objective': 'multi:softmax', 'num_class': 30, 'n_estimators': 1000, 'min_child_weight': 5, 'max_depth': 3, 'learning_rate': 0.1, 'gamma': 1, 'colsample_bytree': 0.6, 'colsample_bynode': 0.6, 'colsample_bylevel': 0.6}
# best_score :  0.9568597560975609
# model_score :  0.9805555555555555
# accuracy_score 0.9805555555555555
# 최적 튠 ACC :  0.9805555555555555
# 걸린신간 :  7.92 초