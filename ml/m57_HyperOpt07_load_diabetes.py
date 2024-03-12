from sklearn.datasets import load_diabetes
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import time
from sklearn.svm import LinearSVC
import warnings
from sklearn.utils import all_estimators
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler, RobustScaler,StandardScaler,MaxAbsScaler, LabelEncoder
from sklearn.utils import all_estimators
from sklearn.ensemble import RandomForestClassifier
import warnings
from xgboost import XGBClassifier, XGBRFRegressor, XGBRegressor
from bayes_opt import BayesianOptimization
import hyperopt
from hyperopt import hp, fmin, tpe, Trials, STATUS_OK

import numpy as np
import time
warnings.filterwarnings ('ignore')

X, y = load_diabetes(return_X_y=True)


# lae = LabelEncoder()
# y = lae.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=698423134)

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



# {'target': 0.4416672388476043, 'params': {'colsample_bytree': 0.9375926069718467,
# 'learning_rate': 0.37467015807222215, 'max_bin': 312.60601187260687,
# 'max_depth': 4.931110203428241, 'min_child_samples': 57.476771000582644,
# 'min_child_weight': 30.248640879191274, 'num_leaves': 31.9099508633321,
# 'reg_alpha': 23.70772749433951, 'reg_lambda': 7.783070240357305, 'subsample': 0.5276578478384795}}
# 100  번 걸린시간 :  9.89 초

# best loss: 0.14746048161545677]
# best :  {'colsample_bytree': 0.5994370651742592, 'learning_rate': 0.9781367300579656,
#          'max_bin': 201.0, 'max_depth': 4.0, 'min_child_samples': 199.0,
#          'min_child_weight': 7.0, 'num_leaves': 25.0, 'reg_alpha': 37.32244400128371,
#          'reg_lambda': 3.0234613257263474, 'subsample': 0.7090021855806923}
# 500  번 걸린시간 :  1.66 초
