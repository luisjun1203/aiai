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
import hyperopt
from hyperopt import hp, fmin, tpe, Trials, STATUS_OK
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

# {'target': 0.9805555555555555, 'params': {'colsample_bytree': 0.996737919888327,
#                                           'learning_rate': 0.45465421715663434, 'max_bin': 37.55984191851621,
#                                           'max_depth': 8.55220097607424, 'min_child_samples': 167.46249622267607,
#                                           'min_child_weight': 4.009419693997248, 'num_leaves': 25.518373033012498,
#                                           'reg_alpha': 1.4875666080107868, 'reg_lambda': 7.968237638905344, 'subsample': 0.6185273321993106}}
# 100  번 걸린시간 :  36.1 초

#  best loss: 0.4
# best :  {'colsample_bytree': 0.5082484385008453, 'learning_rate': 0.10264791912821308,
#          'max_bin': 167.0, 'max_depth': 4.0, 'min_child_samples': 195.0, 'min_child_weight': 6.0,
#          'num_leaves': 38.0, 'reg_alpha': 0.522448170495648, 'reg_lambda': 9.65123640774719, 'subsample': 0.5648600965862974}
# 500  번 걸린시간 :  8.84 초












