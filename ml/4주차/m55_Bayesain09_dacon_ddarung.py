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


bayesian_params = {
    'learning_rate' : (0.001, 1),
    'max_depth' : (3, 10),
    'num_leaves' : (24, 40),
    'min_child_samples' : (10, 200),
    'min_child_weight' : (1, 50),
    'subsample' : (0.5, 1),
    'colsample_bytree' : (0.5, 1),
    'max_bin' : (9, 500),
    'reg_lambda' : (-0.001, 10),
    'reg_alpha' : (0.01, 50),
}


def xgb_hamsu(learning_rate, max_depth, num_leaves, min_child_samples, min_child_weight,
              subsample, colsample_bytree, max_bin, reg_alpha, reg_lambda):

    params = {
        'n_estimators' : 100, 
        'learning_rate' : learning_rate,
        'max_depth' : int(round(max_depth)),                    # 무조건 정수형
        'num_leaves' : int(round(num_leaves)),
        'min_child_samples' : int(round(min_child_samples)),
        'min_child_weight' : int(round(min_child_weight)),
        'subsample' : max(min(subsample, 1), 0),                # 0~1 사이의 값
        'colsample_bytree' : colsample_bytree,
        'max_bin' : max(int(round(max_bin)), 10),               # 무조건 10이상
        'reg_lambda' : max(reg_lambda, 0),                      # 무조건 양수만
        'reg_alpha' : reg_alpha,  
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

bay = BayesianOptimization(
    f = xgb_hamsu,
    pbounds=bayesian_params,
    random_state=3
) 
n_iter = 100
start = time.time()
bay.maximize(init_points=5,
             n_iter=n_iter)
end = time.time()

print(bay.max)
print(n_iter, " 번 걸린시간 : ", round(end - start, 2), "초" )

# {'target': 0.7631006276530861, 'params': {'colsample_bytree': 0.8519555628953133,
# 'learning_rate': 0.07792750512384768, 'max_bin': 216.72170328299686,
# 'max_depth': 7.036213565236839, 'min_child_samples': 40.18898604815691,
# 'min_child_weight': 17.16759148046409, 'num_leaves': 33.78168285793042,
# 'reg_alpha': 23.622509583458626, 'reg_lambda': 5.056525952504295, 'subsample': 0.680595518943103}}
# 100  번 걸린시간 :  12.69 초