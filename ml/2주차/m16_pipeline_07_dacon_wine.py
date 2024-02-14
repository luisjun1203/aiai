# https://dacon.io/competitions/open/235610/data

import numpy as np      # 수치화 연산
import pandas as pd     # 각종 연산 ( 판다스 안의 파일들은 넘파이 형식)
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input
from sklearn.model_selection import train_test_split, KFold, cross_val_score,GridSearchCV, RandomizedSearchCV
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score    # rmse 사용자정의 하기 위해 불러오는것
import time                
from sklearn.utils import all_estimators
import warnings
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import  HalvingGridSearchCV

warnings.filterwarnings('ignore')


#1. 데이터

path = "c:/_data/dacon/wine//"

# print(path + "aaa.csv") # c:/_data/dacon/ddarung/aaa.csv

train_csv = pd.read_csv(path + "train.csv", index_col = 0)  # 인덱스를 컬럼으로 판단하는걸 방지
# \ \\ / // 다 가능
print(train_csv)
test_csv = pd.read_csv(path + "test.csv", index_col = 0)
print(test_csv)
submission_csv = pd.read_csv(path + "sample_submission.csv")   # 여기 있는 id 는 인덱스 취급하지 않는다.
print(submission_csv)

print(train_csv.shape)          # (5497, 13)
print(test_csv.shape)           # (1000, 12) // 아래 서브미션과의 열의 합이 11 인것은 id 열 이 중복되어서이다
print(submission_csv.shape)     # (1000, 2)

print(train_csv.columns)        # 


print(train_csv.info())
print("==========================")
print(test_csv.info())
print(train_csv.describe())

######### 결측치 처리 1. 제거 #########
train_csv = train_csv.dropna()      # 결측치가 한 행에 하나라도 있으면 그 행을 삭제한다
######### 결측치 처리 2. 0으로 #########
# train_csv = train_csv.fillna(0)   # 결측치 행에 0을 집어 넣는다

# print(train_csv.isnull().sum())
print(train_csv.isna().sum())       # 위 와 같다. isnull() = isna()
print(train_csv.info())
print(train_csv.shape)
print(train_csv)

# test_csv = test_csv.fillna(test_csv.mean())     # 널값에 평균을 넣은거
print(test_csv.info())

# type에는 white와 red 두 종류가 있습니다.
# 각각 0,1로 변환합니다.
train_csv['type'] = train_csv['type'].map({'white':0, 'red':1}).astype(int)
test_csv['type'] = test_csv['type'].map({'white':0, 'red':1}).astype(int)


######### x 와 y 를 분리 #########
x = train_csv.drop(['quality'], axis = 1)     # species 를 삭제하는데 count가 열이면 액시스 1, 행이면 0
y = train_csv['quality']

print(np.unique(y, return_counts=True))
# (array([3, 4, 5, 6, 7, 8, 9], dtype=int64), array([  26,  186, 1788, 2416,  924,  152,    5], dtype=int64))




print(pd.value_counts(y))
# 6    2416


x_train, x_test, y_train, y_test = train_test_split(x, y, 
            shuffle=True, train_size= 0.7,
            random_state= 78567,
            stratify=y)    # 스트레티파이 y_ohe2


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
                     , random_state=123,
                    #  n_iter=10
                    factor=3,
                    min_resources=300
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