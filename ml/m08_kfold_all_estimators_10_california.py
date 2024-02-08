from sklearn.datasets import fetch_california_housing
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
from sklearn.model_selection import train_test_split
import time                     # 시간 알고싶을때
from sklearn.svm import LinearSVR
import warnings
from sklearn.utils import all_estimators
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler, RobustScaler,StandardScaler,MaxAbsScaler
from sklearn.utils import all_estimators
import warnings
warnings.filterwarnings ('ignore')

datasets = fetch_california_housing()
X = datasets.data
y = datasets.target
n_splits= 5
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=123)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, shuffle=True, random_state=42)

# print(X_train)
# print(X_test)

# allAlgorithms = all_estimators(type_filter='classifier')
allAlgorithms = all_estimators(type_filter='regressor')

# print("allAlgorithms : ", allAlgorithms)
# print("모델의 갯수 : ", len(allAlgorithms))     # 분류모델의 갯수 :  41, 회귀모델의 갯수 :  55


for name, algorithm in allAlgorithms:
    try:
        #2. 모델 구성
        model = algorithm()
        #3. 컴파일, 훈련
        scores = cross_val_score(model, X_train, y_train, cv=kfold)
        print("==================", name, "=======================")
        print("ACC : ", scores, "\n 평균 ACC : ", round(np.mean(scores), 4))
        # 4. 평가, 예측
        y_predict = cross_val_predict(model, X_test, y_test ,cv=kfold)
        acc = accuracy_score(y_test, y_predict)
        print("cross_val_accuracy : ", acc)
        
    except:
        print(name, ' :은 바보멍충이!!')
        # continue

# epochs=10000, batch_size=80, test_size=0.15, random_state=59
# 로스 :  0.5511764883995056
# R2스코어 :  0.6172541292238007

# mse
# 로스 :  0.6345612406730652
# R2스코어 :  0.5226054954200192
# mae
# 로스 :  0.5319810509681702
# R2스코어 :  0.5684984592213133

# model.score :  0.35430089226834105
# R2스코어 :  0.35430089226834105



# AdaBoostClassifier 의 정답률은 :  0.0
# BaggingClassifier 의 정답률은 :  0.022222222222222223
# BernoulliNB 의 정답률은 :  0.0
# CalibratedClassifierCV 의 정답률은 :  0.022222222222222223
# CategoricalNB 의 정답률은 :  0.0
# ClassifierChain  :은 바보멍충이!!
# ComplementNB  :은 바보멍충이!!
# DecisionTreeClassifier 의 정답률은 :  0.0
# DummyClassifier 의 정답률은 :  0.0
# ExtraTreeClassifier 의 정답률은 :  0.0
# ExtraTreesClassifier 의 정답률은 :  0.022222222222222223
# GaussianNB 의 정답률은 :  0.022222222222222223
# GaussianProcessClassifier 의 정답률은 :  0.0
# GradientBoostingClassifier 의 정답률은 :  0.044444444444444446
# HistGradientBoostingClassifier 의 정답률은 :  0.044444444444444446
# KNeighborsClassifier 의 정답률은 :  0.0
# LabelPropagation 의 정답률은 :  0.0
# LabelSpreading 의 정답률은 :  0.0
# LinearDiscriminantAnalysis 의 정답률은 :  0.044444444444444446
# LinearSVC 의 정답률은 :  0.022222222222222223
# LogisticRegression 의 정답률은 :  0.0
# LogisticRegressionCV  :은 바보멍충이!!
# MLPClassifier 의 정답률은 :  0.0
# MultiOutputClassifier  :은 바보멍충이!!
# MultinomialNB  :은 바보멍충이!!
# NearestCentroid 의 정답률은 :  0.0
# NuSVC  :은 바보멍충이!!
# OneVsOneClassifier  :은 바보멍충이!!
# OneVsRestClassifier  :은 바보멍충이!!
# OutputCodeClassifier  :은 바보멍충이!!
# PassiveAggressiveClassifier 의 정답률은 :  0.0
# Perceptron 의 정답률은 :  0.022222222222222223
# QuadraticDiscriminantAnalysis  :은 바보멍충이!!
# RadiusNeighborsClassifier 의 정답률은 :  0.0
# RandomForestClassifier 의 정답률은 :  0.022222222222222223
# RidgeClassifier 의 정답률은 :  0.022222222222222223
# RidgeClassifierCV 의 정답률은 :  0.0
# SGDClassifier 의 정답률은 :  0.0
# SVC 의 정답률은 :  0.044444444444444446
# StackingClassifier  :은 바보멍충이!!
# TheilSenRegressor 의 정답률은 :  0.2314367079580224
# TransformedTargetRegressor  :은 바보멍충이!!
# TweedieRegressor 의 정답률은 :  0.49164985871407174
# VotingRegressor  :은 바보멍충이!!
