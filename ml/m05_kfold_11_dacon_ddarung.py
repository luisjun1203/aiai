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
from sklearn. svm import SVR
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

test_csv = test_csv.fillna(test_csv.mean())
print(test_csv.info())      # 717 non-null

X = train_csv.drop(['count'], axis=1)
y = train_csv['count']

n_splits= 5
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=123)



# 2.모델
model = SVR()

# 3. 훈련
scores = cross_val_score(model, X, y, cv=kfold)

print("ACC : ", scores, "\n 평균 ACC : ", round(np.mean(scores), 4))


# y_predict = model.predict(X_test)

# y_submit = model.predict(test_csv)

# r2 = r2_score(y_test, y_predict)
# print( "R2 스코어 : ", r2)
# submission_csv['count'] = y_submit

# submission_csv.to_csv(path + "submission_0207_1.csv", index=False)


# ACC :  [0.82296914 0.73777655 0.76490903 0.7886017  0.76187034 0.80576339
#  0.79796407]
#  평균 ACC :  0.7828


# #  random_state=3
# 로스 :  2983.10009765625
# R2스코어 :  0.6315264586114105



# model.score :  0.4966459972210009
# R2 스코어 :  0.4966459972210009



# ARDRegression 의 정답률은 :  0.7157112460020667
# AdaBoostRegressor 의 정답률은 :  0.5270506580567736
# BaggingRegressor 의 정답률은 :  0.8514258099065722
# BayesianRidge 의 정답률은 :  0.7151809662563697
# CCA  :은 바보멍충이!!
# DecisionTreeRegressor 의 정답률은 :  0.6956229360882128
# DummyRegressor 의 정답률은 :  -0.01264282645421777
# ElasticNet 의 정답률은 :  0.707203045607421
# ElasticNetCV 의 정답률은 :  0.6394956955269883
# ExtraTreeRegressor 의 정답률은 :  0.6728953719332519
# ExtraTreesRegressor 의 정답률은 :  0.8762990055712846
# GammaRegressor 의 정답률은 :  0.5473420255490494
# GaussianProcessRegressor 의 정답률은 :  -1.3411089670741565
# GradientBoostingRegressor 의 정답률은 :  0.8662767224589
# HistGradientBoostingRegressor 의 정답률은 :  0.8758552833658662
# HuberRegressor 의 정답률은 :  0.6835299345042267
# IsotonicRegression  :은 바보멍충이!!
# KNeighborsRegressor 의 정답률은 :  0.40247448412422704
# KernelRidge 의 정답률은 :  0.7126141348905266
# Lars 의 정답률은 :  0.7191655987347882
# LarsCV 의 정답률은 :  0.7191655987347882
# Lasso 의 정답률은 :  0.7127328806526763
# LassoCV 의 정답률은 :  0.6933644024833001
# LassoLars 의 정답률은 :  0.28358761681682254
# LassoLarsCV 의 정답률은 :  0.7191655987347882
# LassoLarsIC 의 정답률은 :  0.7163036950447229
# LinearRegression 의 정답률은 :  0.7191655987347895
# LinearSVR 의 정답률은 :  0.5652268978239263
# MLPRegressor 의 정답률은 :  0.7130088409117391
# MultiOutputRegressor  :은 바보멍충이!!
# MultiTaskElasticNet  :은 바보멍충이!!
# MultiTaskElasticNetCV  :은 바보멍충이!!
# MultiTaskLasso  :은 바보멍충이!!
# MultiTaskLassoCV  :은 바보멍충이!!
# NuSVR 의 정답률은 :  0.08484390888697135
# OrthogonalMatchingPursuit 의 정답률은 :  0.44325146780685165
# OrthogonalMatchingPursuitCV 의 정답률은 :  0.6971942774646274
# PLSCanonical  :은 바보멍충이!!
# PLSRegression 의 정답률은 :  0.7130397379731289
# PassiveAggressiveRegressor 의 정답률은 :  0.5198301179283158
# PoissonRegressor 의 정답률은 :  -0.012488683051069094
# QuantileRegressor 의 정답률은 :  0.4626880107396707
# RANSACRegressor 의 정답률은 :  0.6265697204075334
# RadiusNeighborsRegressor  :은 바보멍충이!!
# RandomForestRegressor 의 정답률은 :  0.8639088983666489
# RegressorChain  :은 바보멍충이!!
# Ridge 의 정답률은 :  0.7156822861976403
# RidgeCV 의 정답률은 :  0.718307796780151
# SGDRegressor 의 정답률은 :  -7.041675262549504e+25
# SVR 의 정답률은 :  0.09197948117229648
# StackingRegressor  :은 바보멍충이!!
# TheilSenRegressor 의 정답률은 :  0.6929980693021178
# TransformedTargetRegressor 의 정답률은 :  0.7191655987347895
# TweedieRegressor 의 정답률은 :  0.6771865706362916
# VotingRegressor  :은 바보멍충이!!

















































