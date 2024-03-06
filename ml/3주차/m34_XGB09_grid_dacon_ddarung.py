# https://dacon.io/competitions/open/235576/mysubmission

import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.svm import LinearSVR
import warnings
from sklearn.utils import all_estimators
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler, RobustScaler,StandardScaler,MaxAbsScaler
from sklearn.utils import all_estimators
import warnings
from sklearn.model_selection import StratifiedKFold, cross_val_predict, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
import time
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBClassifier, XGBRegressor

warnings.filterwarnings ('ignore')


# 1.데이터

path = "c:\\_data\\dacon\\ddarung\\"

train_csv = pd.read_csv(path + "train.csv", index_col=0)        # 컬럼 지정         , # index_col = : 지정 안해주면 인덱스도 컬럼 판단

test_csv = pd.read_csv(path + "test.csv", index_col=0)

submission_csv = pd.read_csv(path + "submission.csv")

train_csv['hour_bef_precipitation'] = train_csv['hour_bef_precipitation'].fillna(0)
train_csv['hour_bef_pm10'] = train_csv['hour_bef_pm10'].fillna(0)
train_csv['hour_bef_pm2.5'] = train_csv['hour_bef_pm2.5'].fillna(0)
train_csv['hour_bef_windspeed'] = train_csv['hour_bef_windspeed'].fillna(0)
train_csv['hour_bef_temperature'] = train_csv['hour_bef_temperature'].fillna(train_csv['hour_bef_temperature'].mean())
train_csv['hour_bef_humidity'] = train_csv['hour_bef_humidity'].fillna(train_csv['hour_bef_humidity'].mean())
train_csv['hour_bef_visibility'] = train_csv['hour_bef_visibility'].fillna(train_csv['hour_bef_visibility'].mean())
train_csv['hour_bef_ozone'] = train_csv['hour_bef_ozone'].fillna(train_csv['hour_bef_ozone'].mean())



test_csv['hour_bef_precipitation'] = test_csv['hour_bef_precipitation'].fillna(0)
test_csv['hour_bef_pm10'] = test_csv['hour_bef_pm10'].fillna(0)
test_csv['hour_bef_pm2.5'] = test_csv['hour_bef_pm2.5'].fillna(0)
test_csv['hour_bef_windspeed'] = test_csv['hour_bef_windspeed'].fillna(0)
test_csv['hour_bef_temperature'] = test_csv['hour_bef_temperature'].fillna(test_csv['hour_bef_temperature'].mean())
test_csv['hour_bef_humidity'] = test_csv['hour_bef_humidity'].fillna(test_csv['hour_bef_humidity'].mean())
test_csv['hour_bef_visibility'] = test_csv['hour_bef_visibility'].fillna(test_csv['hour_bef_visibility'].mean())
test_csv['hour_bef_ozone'] = test_csv['hour_bef_ozone'].fillna(test_csv['hour_bef_ozone'].mean())

# test_csv = test_csv.fillna(test_csv.mean())
# print(test_csv.info())      # 717 non-null

X = train_csv.drop(['count'], axis=1)
y = train_csv['count']
n_splits= 5
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=2928297790)





X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, shuffle=True, random_state=698423134)

parameters = {
    'n_estimators': [300],  # 부스팅 라운드의 수/ 디폴트 100/ 1 ~ inf/ 정수
    'learning_rate': [0.05],  # 학습률/ 디폴트 0.3/0~1/
    'max_depth': [9],  # 트리의 최대 깊이/ 디폴트 6/ 0 ~ inf/ 정수
    'min_child_weight':  [5],  # 자식에 필요한 모든 관측치에 대한 가중치 합의 최소/ 디폴트 1 / 0~inf
    'gamma': [1],  # 리프 노드를 추가적으로 나눌지 결정하기 위한 최소 손실 감소/ 디폴트 0/ 0~ inf
    'subsample': [0.6],  # 각 트리마다의 관측 데이터 샘플링 비율/ 디폴트 1 / 0~1
    'colsample_bytree': [0.6],  # 각 트리 구성에 필요한 컬럼(특성) 샘플링 비율/ 디폴트 1 / 0~1
    'colsample_bylevel': [0.6], #  디폴트 1 / 0~1
    'colsample_bynode': [0.6], #  디폴트 1 / 0~1
    'reg_alpha' : [0],   # 디폴트 0 / 0 ~ inf / L1 절대값 가중치 규제(제한) / alpha
    'reg_lambda' : [1],   # 디폴트 1 / 0 ~ inf / L2 제곱 가중치 규제(제한) / lambda
    'objective': ['reg:squarederror'],  # 학습 태스크 파라미터
    # 'num_class': [30],
    'verbosity' : [1] 
}

 #2. 모델 구성
model = GridSearchCV(XGBRegressor(random_state=3608501786), parameters, cv=kfold, verbose=1,
                    # refit = True,     # default
                    #  n_jobs=-1
                     )



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
print("최적튠 r2 : " , r2_score(y_test, y_pred_best))

print("걸린시간 : ", round(end_time - start_time, 2), "초")


y_submit = model.predict(test_csv)

submission_csv['count'] = y_submit




submission_csv.to_csv(path + "submission_0219_1.csv", index=False)
def RMSE(y_test, y_predict):
    np.sqrt(mean_squared_error(y_test, y_predict))
    return np.sqrt(mean_squared_error(y_test, y_predict))
rmse = RMSE(y_test, y_predict)
print("RMSE : ", rmse)



# 최적의 파라미터 :  {'colsample_bylevel': 0.6, 'colsample_bynode': 0.6, 'colsample_bytree': 0.6, 'gamma': 1, 'learning_rate': 0.05, 'max_depth': 9, 'min_child_weight': 5, 'n_estimators': 300, 'objective': 'reg:squarederror', 'reg_alpha': 0, 'reg_lambda': 1, 'subsample': 0.6, 'verbosity': 1}
# best_score :  0.7568789483883945
# model.score :  0.86363466782278
# r2_score :  0.86363466782278
# 최적튠 r2 :  0.86363466782278
# 걸린시간 :  127.4 초
# RMSE :  31.651131622845643

# 최적의 파라미터 :  {'colsample_bylevel': 0.6, 'colsample_bynode': 0.6, 'colsample_bytree': 0.6, 'gamma': 1, 'learning_rate': 0.05, 'max_depth': 9, 'min_child_weight': 5, 'n_estimators': 300, 'objective': 'reg:squarederror', 'reg_alpha': 0, 'reg_lambda': 1, 'subsample': 0.6, 'verbosity': 1}
# best_score :  0.7679522113225545
# model.score :  0.746192437677816
# r2_score :  0.746192437677816
# 최적튠 r2 :  0.746192437677816
# 걸린시간 :  2.02 초
# RMSE :  40.831709886700914



