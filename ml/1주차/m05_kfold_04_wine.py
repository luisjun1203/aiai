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
from sklearn.model_selection import train_test_split, KFold, cross_val_score,StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
warnings.filterwarnings ('ignore')


X, y = load_wine(return_X_y=True)

n_splits= 5
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=123)



# 2.모델
model = RandomForestClassifier()

# 3. 훈련
scores = cross_val_score(model, X, y, cv=kfold)

print("ACC : ", scores, "\n 평균 ACC : ", round(np.mean(scores), 4))

# ACC :  [1.         1.         0.97222222 1.         0.94285714] 
#  평균 ACC :  0.983


# ACC :  [0.94444444 1.         0.94444444 0.97142857 0.97142857] 
#  평균 ACC :  0.9663




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




























