from sklearn.datasets import fetch_covtype
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from keras.utils import to_categorical
from sklearn.preprocessing import OneHotEncoder
from keras. callbacks import EarlyStopping
from sklearn.svm import LinearSVC
import warnings
from sklearn.utils import all_estimators
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler, RobustScaler,StandardScaler,MaxAbsScaler, LabelEncoder
from sklearn.utils import all_estimators
import warnings
import time
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier, XGBRFRegressor
from bayes_opt import BayesianOptimization
import hyperopt
from hyperopt import hp, fmin, tpe, Trials, STATUS_OK

warnings.filterwarnings ('ignore')

datasets = fetch_covtype()

X = datasets.data
y = datasets.target

lae = LabelEncoder()
y= lae.fit_transform(y)

# n_splits= 5
# kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=123)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, shuffle=True, random_state=1226)


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
    return -results

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

# {'target': 0.9541495989811022, 'params': {'colsample_bytree': 0.8534257433516694,
# 'learning_rate': 0.9874191819294652, 'max_bin': 236.26336767410208,
# 'max_depth': 8.627306903083172, 'min_child_samples': 98.56796787435233,
# 'min_child_weight': 46.43150846815228, 'num_leaves': 27.882170935567515,
# 'reg_alpha': 4.064660287656759, 'reg_lambda': 1.0118071115635423, 'subsample': 0.9889241963456253}}
# 100  번 걸린시간 :  1069.56 초



#  best loss: 0.94841829885374]
# best :  {'colsample_bytree': 0.6671417313274457, 'learning_rate': 0.7757659780528792, 
#          'max_bin': 351.0, 'max_depth': 9.0, 'min_child_samples': 120.0,
#          'min_child_weight': 27.0, 'num_leaves': 35.0, 'reg_alpha': 12.58987746899078, 
#          'reg_lambda': 5.132150216290233, 'subsample': 0.9740705428183046}
# 500  번 걸린시간 :  507.12 초



