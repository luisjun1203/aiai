import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_log_error, mean_squared_error
import time
from sklearn.svm import LinearSVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor

path = "c:\\_data\\kaggle\\bike\\"

train_csv = pd.read_csv(path + "train.csv", index_col=0)
test_csv = pd.read_csv(path + "test.csv", index_col=0)
submission_csv = pd.read_csv(path + "samplesubmission.csv")

X = train_csv.drop(['casual', 'registered', 'count'], axis=1)
y = train_csv['count']      


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.123, shuffle=True, random_state=6544)

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



# submission_csv['count'] = y_submit                                        
# print("model.score : ", loss)
# submission_csv.to_csv(path + "submission_0207_1_.csv", index=False)

# print("음수갯수 : ",submission_csv[submission_csv['count']<0].count())      # 0보다 작은 조건의 모든 데이터셋을 세줘

# y_predict = model.predict(X_test)                                           #rmse 구하기
# def RMSLE(y_test, y_predict):
#     np.sqrt(mean_squared_log_error(y_test, y_predict))
#     return np.sqrt(mean_squared_log_error(y_test, y_predict))
# rmsle = RMSLE(y_test, y_predict)

# def RMSE(y_test, y_predict):
#     np.sqrt(mean_squared_error(y_test, y_predict))
#     return np.sqrt(mean_squared_error(y_test, y_predict))
# rmse = RMSE(y_test, y_predict)
# # print("256255음수갯수 : ",submission_csv[submission_csv['count']<0].count())      # 0보다 작은 조건의 모든 데이터셋을 세줘

# print("RMSLE : ", rmsle)
# print("RMSE : ", rmse)



# GradientBoostingRegressor accuracy: 0.3841
# GradientBoostingRegressor : [0.07772516 0.00143903 0.03710264 0.01290316 0.22316061 0.2689847
#  0.35854967 0.02013504]


# 초기 모델 정확도: 0.3468
# 수정된 모델 정확도: 0.2778







