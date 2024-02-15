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
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, StandardScaler, RobustScaler
from sklearn.metrics import r2_score, mean_squared_error,mean_squared_log_error,accuracy_score



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


# n_components = 1, Accuracy: -0.128421659786063
# n_components = 2, Accuracy: 0.2383454069898303
# n_components = 3, Accuracy: 0.3473932704401169
# n_components = 4, Accuracy: 0.34275998528859863
# n_components = 5, Accuracy: 0.3457227055397646
# n_components = 6, Accuracy: 0.3388516310919062
# n_components = 7, Accuracy: 0.3482355060944896
# n_components = 8, Accuracy: 0.3593133570137782


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







