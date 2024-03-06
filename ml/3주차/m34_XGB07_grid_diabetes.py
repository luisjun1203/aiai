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


import numpy as np
import time
warnings.filterwarnings ('ignore')

X, y = load_diabetes(return_X_y=True)

n_splits= 5
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=123)

# lae = LabelEncoder()
# y = lae.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=3, test_size=0.2)



parameters = {
    'n_estimators': [30, 50, 100, 200, 300],  # 부스팅 라운드의 수/ 디폴트 100/ 1 ~ inf/ 정수
    'learning_rate': [0.001, 0.01, 0.05, 0.1],  # 학습률/ 디폴트 0.3/0~1/
    'max_depth': [3, 6],  # 트리의 최대 깊이/ 디폴트 6/ 0 ~ inf/ 정수
    'min_child_weight':  [5],  # 자식에 필요한 모든 관측치에 대한 가중치 합의 최소/ 디폴트 1 / 0~inf
    'gamma': [1],  # 리프 노드를 추가적으로 나눌지 결정하기 위한 최소 손실 감소/ 디폴트 0/ 0~ inf
    'subsample': [0.6],  # 각 트리마다의 관측 데이터 샘플링 비율/ 디폴트 1 / 0~1
    'colsample_bytree': [0.6],  # 각 트리 구성에 필요한 컬럼(특성) 샘플링 비율/ 디폴트 1 / 0~1
    'colsample_bylevel': [0.6], #  디폴트 1 / 0~1
    'colsample_bynode': [0.6], #  디폴트 1 / 0~1
    'reg_alpha' : [0],   # 디폴트 0 / 0 ~ inf / L1 절대값 가중치 규제(제한) / alpha
    'reg_lambda' :   [1],   # 디폴트 1 / 0 ~ inf / L2 제곱 가중치 규제(제한) / lambda
    'objective': ['reg:squarederror'],  # 학습 태스크 파라미터
    # 'num_class': [30],
    'verbosity' : [1] 
}
 #2. 모델 구성
model = GridSearchCV(XGBRegressor(), param_grid=parameters, cv=kfold, verbose=1,
                    # refit = True,     # default
                     n_jobs=-1)




start_time = time.time()
model.fit(X_train, y_train)
end_time = time.time()

print("최적의 매개변수 : ", model.best_estimator_)

print("최적의 파라미터 : ", model.best_params_)

print('best_score : ', model.best_score_)
print('model.score : ', model.score(X_test, y_test))

y_predict = model.predict(X_test)
r2 = r2_score(y_test, y_predict)
print("r2_score : ", r2)

y_pred_best = model.best_estimator_.predict(X_test)
print("최적튠 ACC : " , r2_score(y_test, y_pred_best))

print("걸린시간 : ", round(end_time - start_time, 2), "초")




# 최적의 파라미터 :  {'colsample_bylevel': 0.6, 'colsample_bynode': 0.6, 'colsample_bytree': 0.6, 'gamma': 1, 'learning_rate': 0.05, 'max_depth': 3, 'min_child_weight': 5, 'n_estimators': 200, 'objective': 'reg:squarederror', 'reg_alpha': 0, 'reg_lambda': 1, 'subsample': 0.6, 'verbosity': 1}
# best_score :  0.4830826251999419
# model.score :  0.36153810590096747
# r2_score :  0.36153810590096747
# 최적튠 ACC :  0.36153810590096747
# 걸린시간 :  2.09 초

# import sklearn as sk
# print(sk.__version__)