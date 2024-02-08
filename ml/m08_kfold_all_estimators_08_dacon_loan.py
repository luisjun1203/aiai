import numpy as np
import pandas as pd
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout,Input, Conv2D, MaxPooling2D, Flatten, GlobalAveragePooling2D,concatenate, Reshape
from keras. callbacks import EarlyStopping, ModelCheckpoint
from keras. utils import to_categorical
from sklearn.model_selection import train_test_split 
from sklearn.metrics import f1_score
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler
import time
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import os
from sklearn.svm import LinearSVC
import warnings
from sklearn.utils import all_estimators
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler, RobustScaler,StandardScaler,MaxAbsScaler
from sklearn.utils import all_estimators
import warnings
warnings.filterwarnings ('ignore')



# def save_code_to_file(filename=None):
# if filename is None:
#     # 현재 스크립트의 파일명을 가져와서 확장자를 txt로 변경
#     filename = os.path.splitext(os.path.basename(__file__))[0] + ".txt"
# else:
#     filename = filename + ".txt"
# with open(__file__, "r") as file:
#     code = file.read()

# with open(filename, "w") as file:
#     file.write(code)


path = "c:\\_data\\dacon\\loan\\"

train_csv = pd.read_csv(path + "train.csv", index_col=0)
test_csv = pd.read_csv(path + "test.csv", index_col=0)
submission_csv = pd.read_csv(path + "sample_submission.csv")
  

test_csv.loc[test_csv['대출기간']==' 36 months', '대출기간'] =36
train_csv.loc[train_csv['대출기간']==' 36 months', '대출기간'] =36

test_csv.loc[test_csv['대출기간']==' 60 months', '대출기간'] =60
train_csv.loc[train_csv['대출기간']==' 60 months', '대출기간'] =60

test_csv.loc[test_csv['근로기간']=='3', '근로기간'] ='3 years'
train_csv.loc[train_csv['근로기간']=='3', '근로기간'] ='3 years'
test_csv.loc[test_csv['근로기간']=='1 year','근로기간']='1 years'
train_csv.loc[train_csv['근로기간']=='1 year','근로기간']='1 years'
test_csv.loc[test_csv['근로기간']=='<1 year','근로기간']='< 1 year'
train_csv.loc[train_csv['근로기간']=='<1 year','근로기간']='< 1 year'
test_csv.loc[test_csv['근로기간']=='10+years','근로기간']='10+ years'
train_csv.loc[train_csv['근로기간']=='10+years','근로기간']='10+ years'
train_csv.loc[train_csv['근로기간']=='Unknown', '근로기간']='10+ years'
test_csv.loc[test_csv['근로기간']=='Unknown', '근로기간']='10+ years'
train_csv.value_counts('근로기간')

train_csv.loc[train_csv['주택소유상태']=='ANY', '주택소유상태'] = 'OWN'

test_csv.loc[test_csv['대출목적']=='결혼', '대출목적'] = '기타'

lae = LabelEncoder()

lae.fit(train_csv['주택소유상태'])
train_csv['주택소유상태'] = lae.transform(train_csv['주택소유상태'])
test_csv['주택소유상태'] = lae.transform(test_csv['주택소유상태'])

lae.fit(train_csv['대출목적'])
train_csv['대출목적'] = lae.transform(train_csv['대출목적'])
test_csv['대출목적'] = lae.transform(test_csv['대출목적'])

lae.fit(train_csv['근로기간'])
train_csv['근로기간'] = lae.transform(train_csv['근로기간'])
test_csv['근로기간'] = lae.transform(test_csv['근로기간'])

X = train_csv.drop(['대출등급'], axis=1)
y = train_csv['대출등급']
n_splits= 5
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=123)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, shuffle=True, random_state=42, stratify=y)

mms = MinMaxScaler()
mms.fit(X_train)
X_train = mms.transform(X_train) 
X_test = mms.transform(X_test)
test_csv = mms.transform(test_csv)
allAlgorithms = all_estimators(type_filter='classifier')
# allAlgorithms = all_estimators(type_filter='regressor')

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

# y_predict = model.predict(X_test) 

# y_submit = model.predict(test_csv)  

# y_submit = pd.DataFrame(y_submit)
# submission_csv['대출등급'] = y_submit
# # print(y_submit)

# fs = f1_score(y_test, y_predict, average='macro')
# print("f1_score : ", fs)

# submission_csv.to_csv(path + "submisson_02_07_1_.csv", index=False)

# model.score :  0.42464520595361716
# f1_score :  0.22385580661899354



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











