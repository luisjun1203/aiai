import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_log_error, mean_squared_error
import time
from sklearn.svm import LinearSVR
import warnings
from sklearn.utils import all_estimators
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler, RobustScaler,StandardScaler,MaxAbsScaler
from sklearn.utils import all_estimators
from sklearn.model_selection import StratifiedKFold, cross_val_predict, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
import time
from sklearn.ensemble import RandomForestRegressor
from bayes_opt import BayesianOptimization
from xgboost import XGBRegressor

import warnings
warnings.filterwarnings ('ignore')

path = "c:\\_data\\kaggle\\bike\\"

train_csv = pd.read_csv(path + "train.csv", index_col=0)
test_csv = pd.read_csv(path + "test.csv", index_col=0)
submission_csv = pd.read_csv(path + "samplesubmission.csv")

X = train_csv.drop(['casual', 'registered', 'count'], axis=1)
y = train_csv['count']      


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.123, shuffle=True, random_state=6544)


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

# {'target': 0.41533465544710657, 'params': {'colsample_bytree': 0.9016957778658565,
# 'learning_rate': 0.08949808040851201, 'max_bin': 491.0320225475782,
# 'max_depth': 7.145958203188097, 'min_child_samples': 38.63184987236517,
# 'min_child_weight': 5.842035662445236, 'num_leaves': 29.026265300182423,
# 'reg_alpha': 20.55271022852093, 'reg_lambda': 1.3126316968734901, 'subsample': 0.5336446562876074}}
# 100  번 걸린시간 :  16.67 초
