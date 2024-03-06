from sklearn.datasets import load_digits
from xgboost import XGBClassifier, XGBRegressor
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV, HalvingRandomSearchCV, GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import accuracy_score, r2_score,f1_score, roc_auc_score
import time
import pickle

#1.데이터
X,y = load_digits(return_X_y=True)

print(X.shape)  

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=3, stratify=y)

n_splits= 5
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=123)

mms = MinMaxScaler()
mms.fit(X_train)
X_train = mms.transform(X_train)
X_test = mms.transform(X_test)


parameters = {
    'n_estimators': 4000,  # 부스팅 라운드의 수/ 디폴트 100/ 1 ~ inf/ 정수
    'learning_rate': 0.05,  # 학습률/ 디폴트 0.3/0~1/
    'max_depth': 8,  # 트리의 최대 깊이/ 디폴트 6/ 0 ~ inf/ 정수
    'min_child_weight': 1,  # 자식에 필요한 모든 관측치에 대한 가중치 합의 최소/ 디폴트 1 / 0~inf
    'gamma': 0.1,  # 리프 노드를 추가적으로 나눌지 결정하기 위한 최소 손실 감소/ 디폴트 0/ 0~ inf
    'subsample': 0.6,  # 각 트리마다의 관측 데이터 샘플링 비율/ 디폴트 1 / 0~1
    'colsample_bytree': 0.6,  # 각 트리 구성에 필요한 컬럼(특성) 샘플링 비율/ 디폴트 1 / 0~1
    'colsample_bylevel': 0.6, #  디폴트 1 / 0~1
    'colsample_bynode': 0.6, #  디폴트 1 / 0~1
    'reg_alpha' : 0.5,   # 디폴트 0 / 0 ~ inf / L1 절대값 가중치 규제(제한) / alpha
    'reg_lambda' :   0.7,   # 디폴트 1 / 0 ~ inf / L2 제곱 가중치 규제(제한) / lambda
    
}

model = XGBClassifier()

model.set_params(
    **parameters,
    early_stopping_rounds = 100                
                 
                 )

# 3. 훈련
start = time.time()

model.fit(X_train, y_train, 
          eval_set = [(X_train, y_train),(X_test, y_test)],
          verbose = 1,  # true 디폴트 1 / false 디폴트 0 / verbose = n (과정을 n의배수로 보여줌)
       #   eval_metric = 'rmse',     # 회귀 디폴트
        #   eval_metric = 'mae',     # rmsle, mape, mphe....등등
        #   eval_metric = 'logloss',     # 이진분류 디폴트, ACC
          # eval_metric = 'error'    #  이진분류
          eval_metric = 'mlogloss'     #  다중분류 디폴트, ACC
          #  eval_metric = 'auc'       # 이진, 다중 둘다 가능
        
          )


results = model.score(X_test, y_test)
print("최종점수 : ", results)
y_predict = model.predict(X_test)
# 최종점수 :  0.9694444444444444

############################### pickle ###################################################

path = "c://_data//_save//_pickle_test//"
pickle.dump(model, open(path + "m39_pickle1_save.dat", 'wb'))
















