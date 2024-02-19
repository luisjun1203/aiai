from sklearn.datasets import load_breast_cancer
from xgboost import XGBClassifier, XGBRFRegressor
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

#1.데이터
X,y = load_breast_cancer(return_X_y=True)

print(X.shape)  # (569, 30)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, shuffle=False, random_state=3)

n_splits= 5
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=123)

mms = MinMaxScaler()
mms.fit(X_train)
X_train = mms.transform(X_train)
X_test = mms.transform(X_test)



parameters = {
    'n_estimators': [300],  # 부스팅 라운드의 수
    'learning_rate': [0.05, 0.1],  # 학습률
    'max_depth': [3, 6, 9],  # 트리의 최대 깊이
    'min_child_weight': [1, 5, 10],  # 자식에 필요한 모든 관측치에 대한 가중치 합의 최소
    'gamma': [0.5, 1, 1.5, 2],  # 리프 노드를 추가적으로 나눌지 결정하기 위한 최소 손실 감소
    'subsample': [0.6, 0.8, 1.0],  # 각 트리마다의 관측 데이터 샘플링 비율
    'colsample_bytree': [0.6, 0.8, 1.0],  # 각 트리 구성에 필요한 컬럼(특성) 샘플링 비율
    'objective': ['multi:softmax'],  # 학습 태스크 파라미터
    'num_class': [30],
    'verbosity' : [1] 
}

 #2. 모델 구성
model = GridSearchCV(XGBClassifier(), parameters, cv=kfold, verbose=1,
                    # refit = True,     # default
                     n_jobs=-1)
start = time.time()
model.fit(X_train, y_train)
end = time.time()

print("최적의 매개변수 : ", model.best_estimator_)

print("최적의 파라미터 : ", model.best_params_)

print('best_score : ', model.best_score_)
print('model.score : ', model.score(X_test, y_test))

y_predict = model.predict(X_test)
acc = accuracy_score(y_test, y_predict)
print("accuracy_score : ", acc)

y_pred_best = model.best_estimator_.predict(X_test)
print("최적튠 ACC : " , accuracy_score(y_test, y_pred_best))

print("걸린시간 : ", round(end- start, 2), "초")


# 최적의 파라미터 :  {'colsample_bytree': 0.6, 'gamma': 0.5, 'learning_rate': 0.1, 'max_depth': 3, 'min_child_weight': 5, 'n_estimators': 300, 'num_class': 30, 'objective': 'multi:softmax', 'subsample': 0.6, 'verbosity': 1}
# best_score :  0.9618073316283036
# model.score :  0.9649122807017544
# accuracy_score :  0.9649122807017544
# 최적튠 ACC :  0.9649122807017544
# 걸린시간 :  123.59 초








