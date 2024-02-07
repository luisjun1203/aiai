


from sklearn.datasets import load_boston
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
from sklearn.model_selection import train_test_split
import time
from sklearn.svm import LinearSVR
import warnings
from sklearn.utils import all_estimators
warnings.filterwarnings ('ignore')

datasets = load_boston()

X = datasets.data
y = datasets.target


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, shuffle=True, random_state=20)

# allAlgorithms = all_estimators(type_filter='classifier')
allAlgorithms = all_estimators(type_filter='regressor')

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



# random_state=42 epochs=1000, batch_size=15, test_size=0.15
# 로스 :  14.557331085205078
# R2스코어 :  0.7770829847720614

# random_state=42 epochs=10000, batch_size=15,test_size=0.15
# 로스 :  14.244630813598633
# R2스코어 :  0.7818713632982545

# random_state=42, epochs=1000, batch_size=15
# 로스 :  13.158198356628418
# R2스코어 :  0.7985079556506842

# random_state=20, epochs=1500, batch_size=15
# 로스 :  15.55542278289795
# R2스코어 :  0.80151274445764

# random_state=20, epochs=1500, batch_size=15
# loss = 'mse'
# R2스코어 :  0.793961027822985
# loss = 'mae'
# R2스코어 :  0.7919380058581664

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