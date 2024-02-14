from sklearn.datasets import load_wine
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from keras.utils import to_categorical
from sklearn.preprocessing import OneHotEncoder
from keras. callbacks import EarlyStopping
from sklearn.svm import LinearSVC
import warnings
from sklearn.utils import all_estimators
warnings.filterwarnings ('ignore')
datasets = load_wine()

X = datasets.data
y = datasets.target


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, shuffle=True, random_state=713, stratify=y)

allAlgorithms = all_estimators(type_filter='classifier')
# allAlgorithms = all_estimators(type_filter='regressor')

# print("allAlgorithms : ", allAlgorithms)
# print("모델의 갯수 : ", len(allAlgorithms))     # 분류모델의 갯수 :  41, 회귀모델의 갯수 :  55

for name, algorithm in allAlgorithms:
    try:
        #2. 모델 구성
        model = algorithm()
        #3. 컴파일, 훈련
        model.fit(X_train, y_train)
        acc = model.score(X_test, y_test)
        print(name, '의 정답률은 : ', acc)
    except:
        print(name, ' :은 바보멍충이!!')
        # continue

# 로스 :  0.2964943051338196
# ACC :  0.8888888955116272
# accuracy_score :  0.8888888888888888

# model.score :  0.8888888888888888
# accuracy_score :  0.8888888888888888


# AdaBoostClassifier 의 정답률은 :  0.8518518518518519
# BaggingClassifier 의 정답률은 :  0.8518518518518519
# BernoulliNB 의 정답률은 :  0.4074074074074074
# CalibratedClassifierCV 의 정답률은 :  0.9259259259259259
# CategoricalNB  :은 바보멍충이!!
# ClassifierChain  :은 바보멍충이!!
# ComplementNB 의 정답률은 :  0.6666666666666666
# DecisionTreeClassifier 의 정답률은 :  0.7777777777777778
# DummyClassifier 의 정답률은 :  0.4074074074074074
# ExtraTreeClassifier 의 정답률은 :  0.8888888888888888
# ExtraTreesClassifier 의 정답률은 :  0.9629629629629629
# GaussianNB 의 정답률은 :  1.0
# GaussianProcessClassifier 의 정답률은 :  0.48148148148148145
# GradientBoostingClassifier 의 정답률은 :  0.8888888888888888
# HistGradientBoostingClassifier 의 정답률은 :  0.9629629629629629
# KNeighborsClassifier 의 정답률은 :  0.6666666666666666
# LabelPropagation 의 정답률은 :  0.5925925925925926
# LabelSpreading 의 정답률은 :  0.5925925925925926
# LinearDiscriminantAnalysis 의 정답률은 :  0.9629629629629629
# LinearSVC 의 정답률은 :  0.8888888888888888
# LogisticRegression 의 정답률은 :  0.9259259259259259
# LogisticRegressionCV 의 정답률은 :  0.8518518518518519
# MLPClassifier 의 정답률은 :  0.8148148148148148
# MultiOutputClassifier  :은 바보멍충이!!
# MultinomialNB 의 정답률은 :  0.8148148148148148
# NearestCentroid 의 정답률은 :  0.7037037037037037
# NuSVC 의 정답률은 :  0.7777777777777778
# OneVsOneClassifier  :은 바보멍충이!!
# OneVsRestClassifier  :은 바보멍충이!!
# OutputCodeClassifier  :은 바보멍충이!!
# PassiveAggressiveClassifier 의 정답률은 :  0.6666666666666666
# Perceptron 의 정답률은 :  0.5185185185185185
# QuadraticDiscriminantAnalysis 의 정답률은 :  1.0
# RadiusNeighborsClassifier  :은 바보멍충이!!
# RandomForestClassifier 의 정답률은 :  0.9259259259259259
# RidgeClassifier 의 정답률은 :  1.0
# RidgeClassifierCV 의 정답률은 :  1.0
# SGDClassifier 의 정답률은 :  0.6666666666666666
# SVC 의 정답률은 :  0.6666666666666666
# StackingClassifier  :은 바보멍충이!!
# VotingClassifier  :은 바보멍충이!!




























