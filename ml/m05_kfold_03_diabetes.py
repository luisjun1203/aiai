from sklearn.datasets import load_diabetes
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import time
from sklearn.svm import LinearSVC
import warnings
from sklearn.utils import all_estimators
from sklearn.model_selection import train_test_split, KFold, cross_val_score
import numpy as np
from sklearn.svm import SVC  
warnings.filterwarnings ('ignore')


datasets = load_diabetes()
X = datasets.data
y = datasets.target

n_splits= 5
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=123)



# 2.모델
model = SVC(C=100, random_state=123, verbose=1)

# 3. 훈련
scores = cross_val_score(model, X, y, cv=kfold)

print("ACC : ", scores, "\n 평균 ACC : ", round(np.mean(scores), 4))
# loss = 'mse' , random_state=1226, epochs=50, batch_size=10, test_size=0.1
# 로스 :  2031.322509765625+
# R2스코어 :  0.7246106597957658

# model.score :  0.022222222222222223
# R2스코어 :  0.5699312314693923

# ACC :  [0.01123596 0.         0.01136364 0.01136364 0.        ] 
#  평균 ACC :  0.0068



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
# VotingClassifier  :은 바보멍충이!!


