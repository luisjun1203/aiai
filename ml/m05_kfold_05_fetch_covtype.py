from sklearn.datasets import fetch_covtype
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
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier

warnings.filterwarnings ('ignore')

datasets = fetch_covtype()

X = datasets.data
y = datasets.target


n_splits= 5
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=123)



# 2.모델
model = RandomForestClassifier(verbose=1)

# 3. 훈련
scores = cross_val_score(model, X, y, cv=kfold)

print("ACC : ", scores, "\n 평균 ACC : ", round(np.mean(scores), 4))



# ACC :  [0.95432992 0.95426108 0.95562899 0.95646374 0.95529337] 
#  평균 ACC :  0.9552

# ACC :  [0.95418363 0.95363287 0.9553364  0.95615394 0.95604206] 
#  평균 ACC :  0.9551





# 로스 :  0.6938604116439819
# ACC :  0.701832115650177
# accuracy_score :  0.7018321385850623

# 로스 :  0.5239526033401489
# ACC :  0.7768474221229553
# accuracy_score :  0.7768474135779627



# model.score :  0.5275601780138182
# accuracy_score :  0.5275601780138182



# AdaBoostClassifier 의 정답률은 :  0.5671510412824863
# BaggingClassifier 의 정답률은 :  0.9578126920901872
# BernoulliNB 의 정답률은 :  0.6281478203142288
# CalibratedClassifierCV 의 정답률은 :  0.7070049912714219
# CategoricalNB  :은 바보멍충이!!
# ClassifierChain  :은 바보멍충이!!
# ComplementNB  :은 바보멍충이!!
# DecisionTreeClassifier 의 정답률은 :  0.9334415185267144
# DummyClassifier 의 정답률은 :  0.4876005015858966
# ExtraTreeClassifier 의 정답률은 :  0.8643849425880848
# ExtraTreesClassifier 의 정답률은 :  0.9492955668658257
# GaussianNB 의 정답률은 :  0.45996410218583267
# GaussianProcessClassifier  :은 바보멍충이!!
# GradientBoostingClassifier 의 정답률은 :  0.7716063042462689
# HistGradientBoostingClassifier 의 정답률은 :  0.7786334243072459
# KNeighborsClassifier 의 정답률은 :  0.9650856875906666
# LabelPropagation  :은 바보멍충이!!
# LabelSpreading  :은 바보멍충이!!
# LinearDiscriminantAnalysis 의 정답률은 :  0.6794177669592585
# LinearSVC 의 정답률은 :  0.5237540262103219
# LogisticRegression 의 정답률은 :  0.620624031865457
# LogisticRegressionCV 의 정답률은 :  0.6691844311671707
# MLPClassifier 의 정답률은 :  0.7766172457033267
# MultiOutputClassifier  :은 바보멍충이!!
# MultinomialNB  :은 바보멍충이!!
# NearestCentroid 의 정답률은 :  0.1962626933195643
# NuSVC  :은 바보멍충이!!
# OneVsOneClassifier  :은 바보멍충이!!
# OneVsRestClassifier  :은 바보멍충이!!
# OutputCodeClassifier  :은 바보멍충이!!
# PassiveAggressiveClassifier 의 정답률은 :  0.3960954980207027
# Perceptron 의 정답률은 :  0.5281601140862039
# QuadraticDiscriminantAnalysis 의 정답률은 :  0.08342553662314671
# RadiusNeighborsClassifier  :은 바보멍충이!!
# RandomForestClassifier 의 정답률은 :  0.9502003884831944
# RidgeClassifier 의 정답률은 :  0.7003417668609082
# RidgeClassifierCV 의 정답률은 :  0.7003270143345381
# SGDClassifier 의 정답률은 :  0.5785744142017654
# SVC 의 정답률은 :  0.7119618401317892
# StackingClassifier  :은 바보멍충이!!
# VotingClassifier  :은 바보멍충이!!



