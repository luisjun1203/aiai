import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error
from bayes_opt import BayesianOptimization
import time
import warnings
import hyperopt
from hyperopt import hp, fmin, tpe, Trials, STATUS_OK
warnings.filterwarnings('ignore')

# data
X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X,y,train_size=0.9,random_state=47)

sclaer = MinMaxScaler().fit(X_train)
x_train = sclaer.transform(X_train)
x_test = sclaer.transform(X_test)

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
              eval_metric = 'logloss',
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



# {'target': 0.9649122807017544, 'params': {'colsample_bytree': 0.6025477890249413, 'learning_rate': 0.4824219908332704,
# 'max_bin': 299.6035180349157, 'max_depth': 4.042564250250363, 'min_child_samples': 37.360744222751414,
# 'min_child_weight': 39.47038624641532, 'num_leaves': 28.2123559861627, 'reg_alpha': 11.826114953712333,
# 'reg_lambda': 6.3885433219640575, 'subsample': 0.7152353618498895}}
# 100  번 걸린시간 :  13.2 초

#  best loss: -0.9298245614035088
# best :  {'colsample_bytree': 0.6162948448917287, 'learning_rate': 0.3280793161432899,
#          'max_bin': 324.0, 'max_depth': 6.0, 'min_child_samples': 51.0, 
#          'min_child_weight': 44.0, 'num_leaves': 35.0, 'reg_alpha': 22.448112752639744,
#          'reg_lambda': 6.379288469451773, 'subsample': 0.767545370379666}
# 500  번 걸린시간 :  1.71 초


