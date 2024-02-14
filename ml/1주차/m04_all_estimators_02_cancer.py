import numpy as np
from sklearn.datasets import load_breast_cancer
from keras.models import Sequential
from keras.layers import Dense
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error,mean_squared_log_error,accuracy_score
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, StandardScaler, RobustScaler
from sklearn.svm import LinearSVC
import warnings
from sklearn.utils import all_estimators
warnings.filterwarnings ('ignore')
#1. 데이터
datasets= load_breast_cancer()

X = datasets.data
y = datasets.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, shuffle=False, random_state=3)

mms = MinMaxScaler()
mms.fit(X_train)
X_train = mms.transform(X_train)
X_test = mms.transform(X_test)

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


# MinMaxScaler
# 정확도 :  0.9956140350877193
# 로스 :  [0.016668006777763367, 0.9956140518188477]
# R2 :  0.9760378349973726

# MaxAbsScaler
# 정확도 :  0.956140350877193
# 로스 :  [0.5653559565544128, 0.9561403393745422]
# R2 :  0.7603783499737258

# StandardScaler
# 정확도 :  0.9780701754385965
# 로스 :  [0.303570419549942, 0.9780701994895935]
# R2 :  0.8801891749868629

# # RobustScaler
# 정확도 :  0.9692982456140351
# 로스 :  [0.09083357453346252, 0.969298243522644]
# R2 :  0.832264844981608

# linearsvc
# 정확도 :  0.9649122807017544
# model.score :  0.9649122807017544
# R2 :  0.8083026799789806



# AdaBoostClassifier 의 정답률은 :  0.9605263157894737
# BaggingClassifier 의 정답률은 :  0.9342105263157895
# BernoulliNB 의 정답률은 :  0.2894736842105263
# CalibratedClassifierCV 의 정답률은 :  0.9824561403508771
# CategoricalNB 의 정답률은 :  0.7631578947368421
# ClassifierChain  :은 바보멍충이!!
# ComplementNB 의 정답률은 :  0.8070175438596491
# DecisionTreeClassifier 의 정답률은 :  0.868421052631579
# DummyClassifier 의 정답률은 :  0.7587719298245614
# ExtraTreeClassifier 의 정답률은 :  0.8991228070175439
# ExtraTreesClassifier 의 정답률은 :  0.9780701754385965
# GaussianNB 의 정답률은 :  0.9429824561403509
# GaussianProcessClassifier 의 정답률은 :  0.9868421052631579
# GradientBoostingClassifier 의 정답률은 :  0.956140350877193
# HistGradientBoostingClassifier 의 정답률은 :  0.9605263157894737
# KNeighborsClassifier 의 정답률은 :  0.9605263157894737
# LabelPropagation 의 정답률은 :  0.9868421052631579
# LabelSpreading 의 정답률은 :  0.9868421052631579
# LinearDiscriminantAnalysis 의 정답률은 :  0.9736842105263158
# LinearSVC 의 정답률은 :  0.9736842105263158
# LogisticRegression 의 정답률은 :  0.9868421052631579
# LogisticRegressionCV 의 정답률은 :  0.9692982456140351
# MLPClassifier 의 정답률은 :  0.956140350877193
# MultiOutputClassifier  :은 바보멍충이!!
# MultinomialNB 의 정답률은 :  0.8728070175438597
# NearestCentroid 의 정답률은 :  0.9692982456140351
# NuSVC 의 정답률은 :  0.956140350877193
# OneVsOneClassifier  :은 바보멍충이!!
# OneVsRestClassifier  :은 바보멍충이!!
# OutputCodeClassifier  :은 바보멍충이!!
# PassiveAggressiveClassifier 의 정답률은 :  0.9605263157894737
# Perceptron 의 정답률은 :  0.8070175438596491
# QuadraticDiscriminantAnalysis 의 정답률은 :  0.9385964912280702
# RadiusNeighborsClassifier  :은 바보멍충이!!
# RandomForestClassifier 의 정답률은 :  0.9605263157894737
# RidgeClassifier 의 정답률은 :  0.9824561403508771
# RidgeClassifierCV 의 정답률은 :  0.9824561403508771
# SGDClassifier 의 정답률은 :  0.9780701754385965
# SVC 의 정답률은 :  0.9692982456140351
# StackingClassifier  :은 바보멍충이!!
# VotingClassifier  :은 바보멍충이!!