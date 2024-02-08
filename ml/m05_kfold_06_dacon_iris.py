# https://dacon.io/competitions/open/235610/mysubmission


from sklearn.model_selection import train_test_split, KFold, cross_val_score

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
from sklearn.naive_bayes import GaussianNB
warnings.filterwarnings ('ignore')


path = "c:\\_data\\dacon\\iris\\"

train_csv = pd.read_csv(path + "train.csv", index_col=0)        
                                             
test_csv = pd.read_csv(path + "test.csv", index_col=0)

submission_csv = pd.read_csv(path + "sample_submission.csv")



X = train_csv.drop(['species'], axis=1)
y = train_csv['species']


n_splits= 5
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=123)



# 2.모델
model = GaussianNB()

# 3. 훈련
scores = cross_val_score(model, X, y, cv=kfold)

print("ACC : ", scores, "\n 평균 ACC : ", round(np.mean(scores), 4))


# ACC :  [0.95833333 0.875      0.875      1.         0.95833333] 
#  평균 ACC :  0.9333




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









