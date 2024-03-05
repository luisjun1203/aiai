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
    'n_estimators': 4000,  # 부스팅 라운드의 수/ 디폴트 100/ 1 ~ inf/ 정수
    'learning_rate': 0.05,  # 학습률/ 디폴트 0.3/0~1/
    'max_depth': 8,  # 트리의 최대 깊이/ 디폴트 6/ 0 ~ inf/ 정수
    'min_child_weight': 1,  # 자식에 필요한 모든 관측치에 대한 가중치 합의 최소/ 디폴트 1 / 0~inf
    'gamma': 0.1,  # 리프 노드를 추가적으로 나눌지 결정하기 위한 최소 손실 감소/ 디폴트 0/ 0~ inf
    'subsample': 0.6,  # 각 트리마다의 관측 데이터 샘플링 비율/ 디폴트 1 / 0~1
    'colsample_bytree': 0.6,  # 각 트리 구성에 필요한 컬럼(특성) 샘플링 비율/ 디폴트 1 /9 0~1
    'colsample_bylevel': 0.6, #  디폴트 1 / 0~1
    'colsample_bynode': 0.6, #  디폴트 1 / 0~1
    'reg_alpha' : 0.5,   # 디폴트 0 / 0 ~ inf / L1 절대값 가중치 규제(제한) / alpha
    'reg_lambda' :   0.7,   # 디폴트 1 / 0 ~ inf / L2 제곱 가중치 규제(제한) / lambda
    
}
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=777, test_size=0.2)


# 2. 모델 구성
importance = [] #  모델이 특정 특성 수로 훈련되었을 때의 정확도(accuracy)를 포함

for i in range(X.shape[1], 0, -1):  # X.shape[1] : 열의수(특성), (시작, 끝, step)


    model = XGBRegressor()
    model.set_params(early_stopping_rounds = 10, **parameters)



# 3. 훈련
    model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_test, y_test)],
            verbose=1,
            eval_metric ='mlogloss' 
          )


# 4. 평가, 예측


    y_predict = model.predict(X_test)
    r2 = r2_score(y_test, y_predict)
    
    importance.append((i, r2))
    
    

    if i > 1:  
        
        fI = model.feature_importances_
        least_important = np.argmin(fI)     # fI 배열 내에서 가장 낮은 중요도 값을 가진 특성의 인덱스를 찾는다.
        X_train = np.delete(X_train, least_important, axis=1)
        X_test = np.delete(X_test, least_important, axis=1)
   
for n_features, acc in importance:
    print(f"특성: {n_features}, R2_score: {r2:.4f}")   

best_feature_count, best_r2 = max(importance, key=lambda x: x[1]) # max함수가 importance 리스트 X에 대해 X[1]값인 acc를 반환

print(f"가장 좋았던 r2: {best_r2:.4f}, 해당할 때의 특성 수: {best_feature_count}")

# 특성: 9, R2_score: 0.6313
# 특성: 8, R2_score: 0.6313
# 특성: 7, R2_score: 0.6313
# 특성: 6, R2_score: 0.6313
# 특성: 5, R2_score: 0.6313
# 특성: 4, R2_score: 0.6313
# 특성: 3, R2_score: 0.6313
# 특성: 2, R2_score: 0.6313
# 특성: 1, R2_score: 0.6313
# 가장 좋았던 r2: 0.7178, 해당할 때의 특성 수: 4