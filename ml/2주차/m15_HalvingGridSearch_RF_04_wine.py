import numpy as np
import pandas as pd
from sklearn.datasets import load_wine
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input
from sklearn.model_selection import train_test_split, KFold, cross_val_score,cross_val_predict,GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score
import time
from sklearn.utils import all_estimators
import warnings
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import  HalvingGridSearchCV
warnings.filterwarnings('ignore')




# 1. 데이터
datasets = load_wine()
x = datasets.data
y = datasets.target

print(x.shape, y.shape) # (178, 13) (178,)
print(pd.value_counts(y))
# 1    71
# 0    59
# 2    48
print(y)

print(np.unique(y, return_counts=True))
# (array([0, 1, 2]), array([59, 71, 48], dtype=int64))
# 0 은 59개
# 1 은 71개
# 2 는 48개
print(pd.value_counts(y))
# 1    71
# 0    59
# 2    48
# ============================================
# # ========== 원 핫 인코딩 전처리 ==============
# # 1) 케라스
# from keras.utils import to_categorical
# y_ohe = to_categorical(y)   # [1. 0. 0. ] 으로 표현
# print(y_ohe)
# print(y_ohe.shape)  # (178, 3)

# # 2) 판다스
# y_ohe2 = pd.get_dummies(y)  # [True  False  False] 으로 표현
# print(y_ohe2)
# print(y_ohe2.shape) # (178, 3)

# # 3) 사이킷런
# from sklearn.preprocessing import OneHotEncoder
# ohe = OneHotEncoder()   # (sparse=False)
# y = y.reshape(-1, 1)    # (행, 열) 형태로 재정의 // -1 은 열의 정수값에 따라 알아서 행을 맞추어 재정의하라 
# y_ohe3 = ohe.fit_transform(y).toarray() # // 투어레이 사용하면 위에 스파라스 안씀. 스파라스 사용하면 투어레이 안씀
# print(y_ohe3)
# print(y_ohe3.shape) # (178, 3)

x_train, x_test, y_train, y_test = train_test_split(x, y, 
            shuffle=True, train_size= 0.7,
            random_state= 7777,
            stratify=y,)    # 스트레티파이 와이(예스)는 분류에서만 쓴다, 트레인 사이즈에 따라 줄여주는것


from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler
from sklearn.preprocessing import StandardScaler, RobustScaler

scaler = RobustScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
print(np.min(x_train))  # 
print(np.min(x_test))   # 
print(np.max(x_train))  # 
print(np.max(x_test))   # 


print(np.unique(y_test, return_counts=True))
# (array([0., 1.], dtype=float32), array([108,  54], dtype=int64))

print(x)
print(y)

from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold

n_splits = 5
# kfold = KFold(n_splits=n_splits, shuffle=True, random_state=123)
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=123)


from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

parameters = [
    {'n_estimators':[100,200], 'max_depth':[6,10,12], 'min_samples_leaf':[3,10]},
    {'max_depth':[6,8,10,12], 'min_samples_leaf':[3,5,7,10]},
    {'min_samples_leaf':[3,5,7,10], 'min_samples_split':[2,3,5,10]},
    {'min_samples_leaf':[3,5,7,10], 'min_samples_split':[2,3,5,10]},
    {'min_samples_split':[2,3,5,10]},
    {'n_jobs':[-1,2,4],'min_samples_split':[2,3,5,10]}
]


# 2 모델
# model = SVC(C=1, kernel='linear', degree=3)
print('==============하빙그리드서치 시작==========================')
model = HalvingGridSearchCV(RandomForestClassifier(), 
                     parameters,
                     cv = kfold,
                     verbose=1,
                    #  refit=True # 디폴트 트루 # 한바퀴 돌린후 다시 돌린다
                     n_jobs=3   # 24개의 코어중 3개 사용 / 전부사용 -1
                     , random_state=66,
                    #  n_iter=10
                     factor=2,
                     min_resources=30
                     )


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
# max_resources_: 124
# aggressive_elimination: False
# factor: 2
# ----------
# iter: 0
# n_candidates: 76
# n_resources: 30
# Fitting 5 folds for each of 76 candidates, totalling 380 fits
# ----------
# iter: 1
# n_candidates: 38
# n_resources: 60
# Fitting 5 folds for each of 38 candidates, totalling 190 fits
# ----------
# iter: 2
# n_candidates: 19
# n_resources: 120
# Fitting 5 folds for each of 19 candidates, totalling 95 fits
# 최적의 매개변수 :  RandomForestClassifier(n_jobs=2)
# 최적의 파라미터 :  {'min_samples_split': 2, 'n_jobs': 2}
# best_score :  0.9833333333333334
# model_score :  0.9814814814814815
# accuracy_score 0.9814814814814815
# 최적 튠 ACC :  0.9814814814814815
# 걸린신간 :  13.15 초
# PS C:\Study> 