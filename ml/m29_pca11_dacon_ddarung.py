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
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, StandardScaler, RobustScaler
from sklearn.metrics import r2_score, mean_squared_error,mean_squared_log_error,accuracy_score

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
sts = StandardScaler()
sts.fit(X_train)
X_train = sts.transform(X_train)
X_test = sts.transform(X_test)

n_features = X_train.shape[1]
accuracy_results = {}

for n in range(1, n_features + 1):
    pca = PCA(n_components=n)
    X_train_p = pca.fit_transform(X_train)
    X_test_p = pca.transform(X_test)  

    model = RandomForestRegressor(random_state=3)
    model.fit(X_train_p, y_train)
    y_predict = model.predict(X_test_p)
    r2 = r2_score(y_test, y_predict)
    
   
    accuracy_results[n] = r2
    print(f"n_components = {n}, Accuracy: {r2}")
    
    
# n_components = 1, Accuracy: 0.3990292199741705
# n_components = 2, Accuracy: 0.4911330348060332
# n_components = 3, Accuracy: 0.6241380203867485
# n_components = 4, Accuracy: 0.6840264093771584
# n_components = 5, Accuracy: 0.7647815559570421
# n_components = 6, Accuracy: 0.7801742718950899
# n_components = 7, Accuracy: 0.7981685040534922
# n_components = 8, Accuracy: 0.79697729933822
# n_components = 9, Accuracy: 0.8003178545598657    
    
    
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















































