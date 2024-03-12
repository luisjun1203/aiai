from sklearn.datasets import fetch_california_housing
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
from sklearn.model_selection import train_test_split
import time                     # 시간 알고싶을때
from sklearn.svm import LinearSVR
import warnings
from sklearn.utils import all_estimators
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import accuracy_score,f1_score, r2_score
from sklearn.preprocessing import MinMaxScaler, RobustScaler,StandardScaler,MaxAbsScaler
from sklearn.utils import all_estimators
from sklearn.model_selection import StratifiedKFold, cross_val_predict, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
import time
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBClassifier, XGBRFRegressor, XGBRegressor
from bayes_opt import BayesianOptimization
import hyperopt
from hyperopt import hp, fmin, tpe, Trials, STATUS_OK

import warnings
warnings.filterwarnings ('ignore')

datasets = fetch_california_housing()
X = datasets.data
y = datasets.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, shuffle=True, random_state=42)


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


# {'target': 0.8447564520541189, 'params': {'colsample_bytree': 0.9188473852233094,
# 'learning_rate': 0.17482769171061327, 'max_bin': 379.08995251741436, 'max_depth': 7.368340699359113,
# 'min_child_samples': 183.03593934896384, 'min_child_weight': 15.951458472176606, 
# 'num_leaves': 37.540400872679264, 'reg_alpha': 0.12426232978920843,
# 'reg_lambda': 7.691287903948922, 'subsample': 0.8219869929984844}}
# 100  번 걸린시간 :  22.68 초


# best loss: 0.09683334068351979]
# best :  {'colsample_bytree': 0.7912505816053071, 'learning_rate': 0.001001896459776044,
#          'max_bin': 170.0, 'max_depth': 5.0, 'min_child_samples': 120.0, 
#          'min_child_weight': 42.0, 'num_leaves': 28.0, 'reg_alpha': 33.61997693954599,
#          'reg_lambda': 2.804500348601586, 'subsample': 0.5735613912130991}\
#              500  번 걸린시간 :  5.79 초