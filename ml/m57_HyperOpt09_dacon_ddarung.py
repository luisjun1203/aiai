# https://dacon.io/competitions/open/235576/mysubmission

import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.svm import LinearSVR
import warnings
from sklearn.utils import all_estimators
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler, RobustScaler,StandardScaler,MaxAbsScaler
from sklearn.utils import all_estimators
import warnings
from sklearn.model_selection import StratifiedKFold, cross_val_predict, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
import time
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBClassifier, XGBRegressor
from bayes_opt import BayesianOptimization
import hyperopt
from hyperopt import hp, fmin, tpe, Trials, STATUS_OK
warnings.filterwarnings ('ignore')


# 1.데이터

path = "c:\\_data\\dacon\\ddarung\\"

train_csv = pd.read_csv(path + "train.csv", index_col=0)        # 컬럼 지정         , # index_col = : 지정 안해주면 인덱스도 컬럼 판단

test_csv = pd.read_csv(path + "test.csv", index_col=0)

submission_csv = pd.read_csv(path + "submission.csv")

train_csv['hour_bef_precipitation'] = train_csv['hour_bef_precipitation'].fillna(0)
train_csv['hour_bef_pm10'] = train_csv['hour_bef_pm10'].fillna(0)
train_csv['hour_bef_pm2.5'] = train_csv['hour_bef_pm2.5'].fillna(0)
train_csv['hour_bef_windspeed'] = train_csv['hour_bef_windspeed'].fillna(0)
train_csv['hour_bef_temperature'] = train_csv['hour_bef_temperature'].fillna(train_csv['hour_bef_temperature'].mean())
train_csv['hour_bef_humidity'] = train_csv['hour_bef_humidity'].fillna(train_csv['hour_bef_humidity'].mean())
train_csv['hour_bef_visibility'] = train_csv['hour_bef_visibility'].fillna(train_csv['hour_bef_visibility'].mean())
train_csv['hour_bef_ozone'] = train_csv['hour_bef_ozone'].fillna(train_csv['hour_bef_ozone'].mean())



test_csv['hour_bef_precipitation'] = test_csv['hour_bef_precipitation'].fillna(0)
test_csv['hour_bef_pm10'] = test_csv['hour_bef_pm10'].fillna(0)
test_csv['hour_bef_pm2.5'] = test_csv['hour_bef_pm2.5'].fillna(0)
test_csv['hour_bef_windspeed'] = test_csv['hour_bef_windspeed'].fillna(0)
test_csv['hour_bef_temperature'] = test_csv['hour_bef_temperature'].fillna(test_csv['hour_bef_temperature'].mean())
test_csv['hour_bef_humidity'] = test_csv['hour_bef_humidity'].fillna(test_csv['hour_bef_humidity'].mean())
test_csv['hour_bef_visibility'] = test_csv['hour_bef_visibility'].fillna(test_csv['hour_bef_visibility'].mean())
test_csv['hour_bef_ozone'] = test_csv['hour_bef_ozone'].fillna(test_csv['hour_bef_ozone'].mean())

# test_csv = test_csv.fillna(test_csv.mean())
# print(test_csv.info())      # 717 non-null

X = train_csv.drop(['count'], axis=1)
y = train_csv['count']





X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, shuffle=True, random_state=698423134)

search_space = {
    'learning_rate' : hp.uniform('learning_rate', 0.001, 1),
    'max_depth' : hp.quniform('max_depth', 3, 10, 1),
    'num_leaves' : hp.quniform('num_leaves', 24, 40, 1),
    'min_child_samples' : hp.quniform('min_child_samples', 10, 200, 1),
    'min_child_weight' : hp.quniform('min_child_weight', 1, 50, 1),
    'subsample' : hp.uniform('subsample', 0.5, 1),
    'colsample_bytree' : hp.uniform('colsample_bytree', 0.5, 1),
    'max_bin' : hp.quniform('max_bin', 9, 500, 1),
    'reg_lambda' : hp.uniform('reg_lambda', -0.001, 10),
    'reg_alpha' : hp.uniform('reg_alpha', 0.01, 50),
}


def xgb_hamsu(search_space):

    params = {
        'n_estimators' : 100, 
        'learning_rate' : search_space['learning_rate'],
        'max_depth' : int(search_space['max_depth']),                    # 무조건 정수형
        'num_leaves' : int(search_space['num_leaves']),
        'min_child_samples' : int(search_space['min_child_samples']),
        'min_child_weight' : int(search_space['min_child_weight']),
        'subsample' : max(min(search_space['subsample'], 1), 0),                # 0~1 사이의 값
        'colsample_bytree' : search_space['colsample_bytree'],
        'max_bin' : max(int(search_space['max_bin']), 10),               # 무조건 10이상
        'reg_lambda' : max(search_space['reg_lambda'], 0),                      # 무조건 양수만
        'reg_alpha' : search_space['reg_alpha'],  
}

    model = XGBRegressor(**params, n_jobs = -1)

    model.fit(X_train, y_train,
              eval_set=[(X_train, y_train), (X_test, y_test)],
              eval_metric = 'rmse',
              verbose = 0,
              early_stopping_rounds = 50
              )
    
    y_predict = model.predict(X_test)
    results = r2_score(y_test, y_predict)
    return results

trial_val = Trials()

n_iter = 500
start = time.time()

best = fmin(fn= xgb_hamsu,
            space=search_space,
            algo=tpe.suggest,       # 알고리즘, 디폴트
            max_evals=50,           # 서치 횟수
            trials=trial_val,
            rstate=np.random.default_rng(seed=10)       # 난수생성,  직접 찾아봐
            # rstate=333,
) 

end = time.time()

print("best : ", best)
print(n_iter, " 번 걸린시간 : ", round(end - start, 2), "초" )


# {'target': 0.7631006276530861, 'params': {'colsample_bytree': 0.8519555628953133,
# 'learning_rate': 0.07792750512384768, 'max_bin': 216.72170328299686,
# 'max_depth': 7.036213565236839, 'min_child_samples': 40.18898604815691,
# 'min_child_weight': 17.16759148046409, 'num_leaves': 33.78168285793042,
# 'reg_alpha': 23.622509583458626, 'reg_lambda': 5.056525952504295, 'subsample': 0.680595518943103}}
# 100  번 걸린시간 :  12.69 초

# best loss: 0.6312109865528853]
# best :  {'colsample_bytree': 0.9785439094572744, 'learning_rate': 0.9198743522359428, 
#          'max_bin': 284.0, 'max_depth': 5.0, 'min_child_samples': 45.0, 
#          'min_child_weight': 42.0, 'num_leaves': 38.0, 'reg_alpha': 12.186036175158648,
#          'reg_lambda': 2.072958693495802, 'subsample': 0.5345089421534707}
# 500  번 걸린시간 :  2.59 초







