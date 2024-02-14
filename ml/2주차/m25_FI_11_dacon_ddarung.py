# https://dacon.io/competitions/open/235576/mysubmission

import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.svm import LinearSVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor


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

test_csv = test_csv.fillna(test_csv.mean())
print(test_csv.info())      # 717 non-null

X = train_csv.drop(['count'], axis=1)
y = train_csv['count']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, shuffle=True, random_state=713)

# 2.모델구성
model = RandomForestRegressor()

model.fit(X_train, y_train)

initial_predictions = model.predict(X_test)
initial_accuracy = r2_score(y_test, initial_predictions)
print(f"초기 모델 정확도: {initial_accuracy:.4f}")

# 특성 중요도 기반으로 하위 20% 컬럼 제거
feature_importances = model.feature_importances_
indices = np.argsort(feature_importances)
bottom_20_percent = int(len(indices) * 0.2)
columns_to_drop = X.columns[indices[:bottom_20_percent]]
X_dropped = X.drop(columns=columns_to_drop)


X_train_dropped, X_test_dropped, y_train, y_test = train_test_split(X_dropped, y, test_size=0.3, random_state=42)

# 수정된 데이터셋으로 모델 재학습
model.fit(X_train_dropped, y_train)

# 수정된 모델 성능 평가
new_predictions = model.predict(X_test_dropped)
new_accuracy = r2_score(y_test, new_predictions)
print(f"수정된 모델 정확도: {new_accuracy:.4f}")
# models = [
# DecisionTreeRegressor(),
# RandomForestRegressor(),
# GradientBoostingRegressor(),
# XGBRegressor()
# ]


# for model in models:
#     model_name = model.__class__.__name__
#     model.fit(X_train, y_train)
#     predictions = model.predict(X_test)
#     r2 = r2_score(y_test, predictions)
#     print(f"{model_name} accuracy: {r2:.4f}")
#     print(model.__class__.__name__, ":", model.feature_importances_)

# y_predict = model.predict(X_test)

# y_submit = model.predict(test_csv)

# # r2 = r2_score(y_test, y_predict)
# # print( "R2 스코어 : ", r2)
# submission_csv['count'] = y_submit

# submission_csv.to_csv(path + "submission_0214_1.csv", index=False)

#  random_state=3
# 로스 :  2983.10009765625
# R2스코어 :  0.6315264586114105



# model.score :  0.4966459972210009
# R2 스코어 :  0.4966459972210009

# RandomForestRegressor accuracy: 0.8696
# RandomForestRegressor : [0.5791259  0.18995091 0.0180268  0.03284095 0.03884749 0.03488213
#  0.04252761 0.03798936 0.02580886]



# 초기 모델 정확도: 0.8775
# 수정된 모델 정확도: 0.7867















































