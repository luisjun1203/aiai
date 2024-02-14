import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_log_error, mean_squared_error
import time
from sklearn.svm import LinearSVR
import warnings
from sklearn.utils import all_estimators
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import MinMaxScaler, RobustScaler,StandardScaler,MaxAbsScaler
from sklearn.ensemble import HistGradientBoostingRegressor


warnings.filterwarnings ('ignore')

path = "c:\\_data\\kaggle\\bike\\"

train_csv = pd.read_csv(path + "train.csv", index_col=0)
test_csv = pd.read_csv(path + "test.csv", index_col=0)
submission_csv = pd.read_csv(path + "samplesubmission.csv")

X = train_csv.drop(['casual', 'registered', 'count'], axis=1)
y = train_csv['count']      


X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=3, test_size=0.2)


scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


n_splits= 5
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=123)



# 2.모델
model = HistGradientBoostingRegressor()

# 3. 훈련
scores = cross_val_score(model, X, y, cv=kfold)

print("ACC : ", scores, "\n 평균 ACC : ", round(np.mean(scores), 4))
# y_submit = model.predict(test_csv)
# # print(y_submit.shape)   
# # r2 = r2_score(y_test, y_submit)


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




# ACC :  [0.33787554 0.34673456 0.35355217 0.38048027 0.37563452] 
#  평균 ACC :  0.3589








# ARDRegression 의 정답률은 :  0.30887130761960313
# AdaBoostRegressor 의 정답률은 :  0.24696015093891044
# BaggingRegressor 의 정답률은 :  0.307162527620503
# BayesianRidge 의 정답률은 :  0.30889229576364996
# CCA  :은 바보멍충이!!
# DecisionTreeRegressor 의 정답률은 :  -0.08949770127002266
# DummyRegressor 의 정답률은 :  -0.0010700171817252802
# ElasticNet 의 정답률은 :  0.30647077016203916
# ElasticNetCV 의 정답률은 :  0.3031374004458536
# ExtraTreeRegressor 의 정답률은 :  0.038368144383383984
# ExtraTreesRegressor 의 정답률은 :  0.28464207515420326
# GammaRegressor 의 정답률은 :  0.20529377559234796
# GaussianProcessRegressor 의 정답률은 :  -0.13737716352606566
# GradientBoostingRegressor 의 정답률은 :  0.38400786661875863
# HistGradientBoostingRegressor 의 정답률은 :  0.4075821965458051
# HuberRegressor 의 정답률은 :  0.28978715970960456
# IsotonicRegression  :은 바보멍충이!!
# KNeighborsRegressor 의 정답률은 :  0.28806831509392106
# KernelRidge 의 정답률은 :  0.2931936157442179
# Lars 의 정답률은 :  0.3089325724639751
# LarsCV 의 정답률은 :  0.3083201550974195
# Lasso 의 정답률은 :  0.3086703535053241
# LassoCV 의 정답률은 :  0.3086383820893682
# LassoLars 의 정답률은 :  -0.0010700171817252802
# LassoLarsCV 의 정답률은 :  0.3083201550974195
# LassoLarsIC 의 정답률은 :  0.30865576734767486
# LinearRegression 의 정답률은 :  0.308932572463975
# LinearSVR 의 정답률은 :  0.27866590686104487
# MLPRegressor 의 정답률은 :  0.3405076673191888
# MultiOutputRegressor  :은 바보멍충이!!
# MultiTaskElasticNet  :은 바보멍충이!!
# MultiTaskElasticNetCV  :은 바보멍충이!!
# MultiTaskLasso  :은 바보멍충이!!
# MultiTaskLassoCV  :은 바보멍충이!!
# NuSVR 의 정답률은 :  0.2640088685603562
# OrthogonalMatchingPursuit 의 정답률은 :  0.17494071841316172
# OrthogonalMatchingPursuitCV 의 정답률은 :  0.30443624060050256
# PLSCanonical  :은 바보멍충이!!
# PLSRegression  :은 바보멍충이!!
# PassiveAggressiveRegressor 의 정답률은 :  -0.21712298980918376
# PoissonRegressor 의 정답률은 :  0.3131634960096831
# QuantileRegressor  :은 바보멍충이!!
# RANSACRegressor  :은 바보멍충이!!
# RadiusNeighborsRegressor  :은 바보멍충이!!
# RandomForestRegressor 의 정답률은 :  0.3552229673099193
# RegressorChain  :은 바보멍충이!!
# Ridge  :은 바보멍충이!!
# RidgeCV  :은 바보멍충이!!
# SGDRegressor 의 정답률은 :  -1.9376416131693308e+16
# SVR 의 정답률은 :  0.2520165447500744
# StackingRegressor  :은 바보멍충이!!
# Breakdown point: 0.0740631737186066
# TheilSenRegressor 의 정답률은 :  0.3064211138774464
# TransformedTargetRegressor  :은 바보멍충이!!
# TweedieRegressor 의 정답률은 :  0.3041184273724099
# VotingRegressor  :은 바보멍충이!!









