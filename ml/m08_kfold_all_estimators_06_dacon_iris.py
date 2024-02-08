# https://dacon.io/competitions/open/235610/mysubmission


from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler, RobustScaler,StandardScaler,MaxAbsScaler
from sklearn.utils import all_estimators
import warnings
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_absolute_error, accuracy_score
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import OneHotEncoder
from keras.utils import to_categorical
from sklearn.svm import LinearSVC 
import warnings
from sklearn.utils import all_estimators
warnings.filterwarnings ('ignore')


path = "c:\\_data\\dacon\\iris\\"

train_csv = pd.read_csv(path + "train.csv", index_col=0)        
                                             
test_csv = pd.read_csv(path + "test.csv", index_col=0)

submission_csv = pd.read_csv(path + "sample_submission.csv")



X = train_csv.drop(['species'], axis=1)
y = train_csv['species']
n_splits= 5
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=123)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, shuffle=True, random_state=1226)           # 1226 713

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
# y_submit = model.predict(test_csv)  
# y_predict = model.predict(X_test) 

# # y_test = np.argmax(y_test, axis=1)
# # y_submit = np.argmax(y_submit, axis=1)
# submission_csv['species'] = y_submit       
                    

# submission_csv.to_csv(path + "submission_02_07_1_.csv", index=False)
# # print(submission_csv)
# acc = accuracy_score(y_test, y_predict)
# print("accuracy_score : ", acc)
    
# model.score :  0.9583333333333334
# accuracy_score :  0.9583333333333334

# AdaBoostClassifier 의 정답률은 :  0.9166666666666666
# BaggingClassifier 의 정답률은 :  0.9166666666666666
# BernoulliNB 의 정답률은 :  0.3333333333333333
# CalibratedClassifierCV 의 정답률은 :  0.875
# CategoricalNB 의 정답률은 :  0.9166666666666666
# ClassifierChain  :은 바보멍충이!!
# ComplementNB 의 정답률은 :  0.6666666666666666
# DecisionTreeClassifier 의 정답률은 :  0.8958333333333334
# DummyClassifier 의 정답률은 :  0.3333333333333333
# ExtraTreeClassifier 의 정답률은 :  0.9583333333333334
# ExtraTreesClassifier 의 정답률은 :  0.9166666666666666
# GaussianNB 의 정답률은 :  0.9583333333333334
# GaussianProcessClassifier 의 정답률은 :  0.9375
# GradientBoostingClassifier 의 정답률은 :  0.9166666666666666
# HistGradientBoostingClassifier 의 정답률은 :  0.9375
# KNeighborsClassifier 의 정답률은 :  0.9166666666666666
# LabelPropagation 의 정답률은 :  0.9166666666666666
# LabelSpreading 의 정답률은 :  0.9166666666666666
# LinearDiscriminantAnalysis 의 정답률은 :  0.9583333333333334
# LinearSVC 의 정답률은 :  0.9166666666666666
# LogisticRegression 의 정답률은 :  0.9375
# LogisticRegressionCV 의 정답률은 :  0.9583333333333334
# MLPClassifier 의 정답률은 :  0.9166666666666666
# MultiOutputClassifier  :은 바보멍충이!!
# MultinomialNB 의 정답률은 :  0.9166666666666666
# NearestCentroid 의 정답률은 :  0.875
# NuSVC 의 정답률은 :  0.9375
# OneVsOneClassifier  :은 바보멍충이!!
# OneVsRestClassifier  :은 바보멍충이!!
# OutputCodeClassifier  :은 바보멍충이!!
# PassiveAggressiveClassifier 의 정답률은 :  0.9166666666666666
# Perceptron 의 정답률은 :  0.5625
# QuadraticDiscriminantAnalysis 의 정답률은 :  0.9583333333333334
# RadiusNeighborsClassifier 의 정답률은 :  0.9166666666666666
# RandomForestClassifier 의 정답률은 :  0.8958333333333334
# RidgeClassifier 의 정답률은 :  0.7916666666666666
# RidgeClassifierCV 의 정답률은 :  0.7916666666666666
# SGDClassifier 의 정답률은 :  0.6666666666666666
# SVC 의 정답률은 :  0.9375
# StackingClassifier  :은 바보멍충이!!
# VotingClassifier  :은 바보멍충이!!









