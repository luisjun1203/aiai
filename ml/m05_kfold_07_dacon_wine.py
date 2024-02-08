# https://dacon.io/competitions/open/235610/mysubmission

from sklearn.model_selection import train_test_split, KFold, cross_val_score

import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from keras.utils import to_categorical
from sklearn.svm import LinearSVC
import warnings
from sklearn.utils import all_estimators
from sklearn.ensemble import RandomForestClassifier
warnings.filterwarnings ('ignore')

path = "C:\\_data\\dacon\\wine\\"

train_csv = pd.read_csv(path + "train.csv", index_col=0)       
# train_csv.to_csv(path + "train_123_csv", index=False)                                             
test_csv = pd.read_csv(path + "test.csv", index_col=0)

submission_csv = pd.read_csv(path + "sample_submission.csv")


lae = LabelEncoder()
lae.fit(train_csv['type'])
train_csv['type'] = lae.transform(train_csv['type'])
test_csv['type'] = lae.transform(test_csv['type'])

X = train_csv.drop(['quality'], axis=1)

y = train_csv['quality']

n_splits= 5
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=123)



# 2.모델
model = RandomForestClassifier()

# 3. 훈련
scores = cross_val_score(model, X, y, cv=kfold)

print("ACC : ", scores, "\n 평균 ACC : ", round(np.mean(scores), 4))


# ACC :  [0.65181818 0.70090909 0.66151046 0.66696997 0.68789809] 
#  평균 ACC :  0.6738


# y_submit = model.predict(test_csv)  
# y_predict = model.predict(X_test) 

# # y_test = np.argmax(y_test, axis=1)
# # y_predict = np.argmax(y_predict, axis=1)
# # y_submit = np.argmax(y_submit, axis=1)+3

# submission_csv['quality'] = y_submit



# submission_csv.to_csv(path + "submission_0207_1_.csv", index=False)

# acc = accuracy_score(y_test, y_predict)
# print("accuracy_score : ", acc)


# model.score :  0.4961346066393815
# accuracy_score :  0.4961346066393815

# AdaBoostClassifier 의 정답률은 :  0.36243747157799
# BaggingClassifier 의 정답률은 :  0.6093678944974988
# BernoulliNB 의 정답률은 :  0.44565711687130516
# CalibratedClassifierCV 의 정답률은 :  0.5079581628012733
# CategoricalNB  :은 바보멍충이!!
# ClassifierChain  :은 바보멍충이!!
# ComplementNB 의 정답률은 :  0.32833105957253295
# DecisionTreeClassifier 의 정답률은 :  0.5466120964074579
# DummyClassifier 의 정답률은 :  0.43974533879035926
# ExtraTreeClassifier 의 정답률은 :  0.5488858572078218
# ExtraTreesClassifier 의 정답률은 :  0.6407457935425194
# GaussianNB 의 정답률은 :  0.42155525238744884
# GaussianProcessClassifier 의 정답률은 :  0.5152341973624375
# GradientBoostingClassifier 의 정답률은 :  0.5852660300136425
# HistGradientBoostingClassifier 의 정답률은 :  0.6330150068212824
# KNeighborsClassifier 의 정답률은 :  0.4483856298317417
# LabelPropagation 의 정답률은 :  0.5084129149613461
# LabelSpreading 의 정답률은 :  0.5084129149613461
# LinearDiscriminantAnalysis 의 정답률은 :  0.5266030013642565
# LinearSVC 의 정답률은 :  0.44247385175079584
# LogisticRegression 의 정답률은 :  0.4679399727148704
# LogisticRegressionCV 의 정답률은 :  0.5206912232833106
# MLPClassifier 의 정답률은 :  0.4579354251932697
# MultiOutputClassifier  :은 바보멍충이!!
# MultinomialNB 의 정답률은 :  0.3501591632560255
# NearestCentroid 의 정답률은 :  0.13824465666211916
# NuSVC  :은 바보멍충이!!
# OneVsOneClassifier  :은 바보멍충이!!
# OneVsRestClassifier  :은 바보멍충이!!
# OutputCodeClassifier  :은 바보멍충이!!
# PassiveAggressiveClassifier 의 정답률은 :  0.47203274215552526
# Perceptron 의 정답률은 :  0.08685766257389722
# QuadraticDiscriminantAnalysis 의 정답률은 :  0.46612096407457937
# RadiusNeighborsClassifier  :은 바보멍충이!!
# RandomForestClassifier 의 정답률은 :  0.6475670759436107
# RidgeClassifier 의 정답률은 :  0.5302410186448385
# RidgeClassifierCV 의 정답률은 :  0.5302410186448385
# SGDClassifier 의 정답률은 :  0.4733969986357435
# SVC 의 정답률은 :  0.43974533879035926
# StackingClassifier  :은 바보멍충이!!
# VotingClassifier  :은 바보멍충이!!











