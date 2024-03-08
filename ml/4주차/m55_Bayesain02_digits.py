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
from sklearn.metrics import accuracy_score
import time
from bayes_opt import BayesianOptimization
import warnings
warnings.filterwarnings('ignore')

# 1 데이터
datasets = load_digits()

X = datasets.data
y = datasets.target

# print(x)
# print(y)    # [0 1 2 ... 8 9 8]
# print(x.shape)  # (1797, 64)    # 64니까 8 * 8
# print(y.shape)  # (1797,)
# print(pd.value_counts(y, sort=False))


X_train, X_test, y_train, y_test = train_test_split(
    X, y, shuffle=True, random_state=123, train_size=0.8, stratify=y
    )

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

    model = XGBClassifier(**params, n_jobs = -1)

    model.fit(X_train, y_train,
              eval_set=[(X_train, y_train), (X_test, y_test)],
              eval_metric = 'mlogloss',
              verbose = 0,
              early_stopping_rounds = 50
              )

    y_predict = model.predict(X_test)
    results = accuracy_score(y_test, y_predict)
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

# {'target': 0.9805555555555555, 'params': {'colsample_bytree': 0.996737919888327,
#                                           'learning_rate': 0.45465421715663434, 'max_bin': 37.55984191851621,
#                                           'max_depth': 8.55220097607424, 'min_child_samples': 167.46249622267607,
#                                           'min_child_weight': 4.009419693997248, 'num_leaves': 25.518373033012498,
#                                           'reg_alpha': 1.4875666080107868, 'reg_lambda': 7.968237638905344, 'subsample': 0.6185273321993106}}
# 100  번 걸린시간 :  36.1 초















