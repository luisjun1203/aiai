import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error
from bayes_opt import BayesianOptimization
import time
import warnings
warnings.filterwarnings('ignore')

# data
X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X,y,train_size=0.9,random_state=47)

sclaer = MinMaxScaler().fit(X_train)
x_train = sclaer.transform(X_train)
x_test = sclaer.transform(X_test)

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
              eval_metric = 'logloss',
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

# {'target': 0.9649122807017544, 'params': {'colsample_bytree': 0.6025477890249413, 'learning_rate': 0.4824219908332704,
# 'max_bin': 299.6035180349157, 'max_depth': 4.042564250250363, 'min_child_samples': 37.360744222751414,
# 'min_child_weight': 39.47038624641532, 'num_leaves': 28.2123559861627, 'reg_alpha': 11.826114953712333,
# 'reg_lambda': 6.3885433219640575, 'subsample': 0.7152353618498895}}
# 100  번 걸린시간 :  13.2 초





