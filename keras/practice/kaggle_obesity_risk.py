import numpy as np
import pandas as pd
import time
from keras.models import Sequential, Model, load_model
from keras. layers import Dense, Conv1D, SimpleRNN, LSTM, Flatten, GRU, Dropout, Input, concatenate
from keras. callbacks import EarlyStopping, ModelCheckpoint
from keras. utils import to_categorical
from sklearn.model_selection import train_test_split 
from sklearn.metrics import f1_score, mean_squared_error, accuracy_score, r2_score
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import keras
import tensorflow as tf
import random as rn
from sklearn.ensemble import RandomForestClassifier 
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict, GridSearchCV, KFold


path = "c:\\_data\\kaggle\\obesity_risk\\"


train_csv = pd.read_csv(path + "train.csv", index_col=0)
test_csv = pd.read_csv(path + "test.csv", index_col=0)
submission_csv = pd.read_csv(path + "sample_submission.csv")



# print(train_csv.shape)      # (20758, 17)
# print(test_csv.shape)       # (13840, 16)
# print(submission_csv.shape) # (13840, 1)

# print(train_csv.info())
#  #   Column                          Non-Null Count  Dtype
# ---  ------                          --------------  -----
#  0   Gender                          20758 non-null  object
#  1   Age                             20758 non-null  float64
#  2   Height                          20758 non-null  float64
#  3   Weight                          20758 non-null  float64
#  4   family_history_with_overweight  20758 non-null  object
#  5   FAVC                            20758 non-null  object
#  6   FCVC                            20758 non-null  float64
#  7   NCP                             20758 non-null  float64
#  8   CAEC                            20758 non-null  object
#  9   SMOKE                           20758 non-null  object
#  10  CH2O                            20758 non-null  float64
#  11  SCC                             20758 non-null  object
#  12  FAF                             20758 non-null  float64
#  13  TUE                             20758 non-null  float64
#  14  CALC                            20758 non-null  object
#  15  MTRANS                          20758 non-null  object
#  16  NObeyesdad                      20758 non-null  object
# dtypes: float64(8), object(9)
# memory usage: 2.9+ MB
# None
# print(test_csv.info())

#  #   Column                          Non-Null Count  Dtype
# ---  ------                          --------------  -----
#  0   Gender                          13840 non-null  object
#  1   Age                             13840 non-null  float64
#  2   Height                          13840 non-null  float64
#  3   Weight                          13840 non-null  float64
#  4   family_history_with_overweight  13840 non-null  object
#  5   FAVC                            13840 non-null  object
#  6   FCVC                            13840 non-null  float64
#  7   NCP                             13840 non-null  float64
#  8   CAEC                            13840 non-null  object
#  9   SMOKE                           13840 non-null  object
#  10  CH2O                            13840 non-null  float64
#  11  SCC                             13840 non-null  object
#  12  FAF                             13840 non-null  float64
#  13  TUE                             13840 non-null  float64
#  14  CALC                            13840 non-null  object
#  15  MTRANS                          13840 non-null  object
# dtypes: float64(8), object(8)
# memory usage: 1.8+ MB
# None

# print(train_csv.isnull().sum())
# print(test_csv.isnull().sum())
########### 결측치 확인 ##################
# Gender                            0
# Age                               0
# Height                            0
# Weight                            0
# family_history_with_overweight    0
# FAVC                              0
# FCVC                              0
# NCP                               0
# CAEC                              0
# SMOKE                             0
# CH2O                              0
# SCC                               0
# FAF                               0
# TUE                               0
# CALC                              0
# MTRANS                            0
# NObeyesdad                        0
# dtype: int64

# Gender                            0
# Age                               0
# Height                            0
# Weight                            0
# family_history_with_overweight    0
# FAVC                              0
# FCVC                              0
# NCP                               0
# CAEC                              0
# SMOKE                             0
# CH2O                              0
# SCC                               0
# FAF                               0
# TUE                               0
# CALC                              0
# MTRANS                            0
# dtype: int64

# print(np.unique(train_csv['Gender'], return_counts=True))
# # (array(['Female', 'Male'], dtype=object), array([10422, 10336], dtype=int64))
# print(np.unique(test_csv['Gender'], return_counts=True))
# # (array(['Female', 'Male'], dtype=object), array([6965, 6875], dtype=int64))

# print(np.unique(train_csv['Age'], return_counts=True))

# print(np.unique(train_csv['NObeyesdad'],return_counts=True))        # y
# (array(['Insufficient_Weight', 'Normal_Weight', 'Obesity_Type_I',
#        'Obesity_Type_II', 'Obesity_Type_III', 'Overweight_Level_I',
#        'Overweight_Level_II'], dtype=object), array([2523, 3082, 2910, 3248, 4046, 2427, 2522], dtype=int64))


# print(np.unique(train_csv['CALC']))
# print(np.unique(test_csv['CALC']))
# ['Frequently' 'Sometimes' 'no']
# ['Always' 'Frequently' 'Sometimes' 'no']

test_csv.loc[test_csv['CALC']=='Always', 'CALC'] = 'Frequently'


lae = LabelEncoder()
lae.fit(train_csv['Gender'])
train_csv['Gender'] = lae.transform(train_csv['Gender'])
test_csv['Gender'] = lae.transform(test_csv['Gender'])



lae.fit(train_csv['family_history_with_overweight'])
train_csv['family_history_with_overweight'] = lae.transform(train_csv['family_history_with_overweight'])
test_csv['family_history_with_overweight'] = lae.transform(test_csv['family_history_with_overweight'])


lae.fit(train_csv['FAVC'])
train_csv['FAVC'] = lae.transform(train_csv['FAVC'])
test_csv['FAVC'] = lae.transform(test_csv['FAVC'])


lae.fit(train_csv['CAEC'])
train_csv['CAEC'] = lae.transform(train_csv['CAEC'])
test_csv['CAEC'] = lae.transform(test_csv['CAEC'])


lae.fit(train_csv['SMOKE'])
train_csv['SMOKE'] = lae.transform(train_csv['SMOKE'])
test_csv['SMOKE'] = lae.transform(test_csv['SMOKE'])

lae.fit(train_csv['SCC'])
train_csv['SCC'] = lae.transform(train_csv['SCC'])
test_csv['SCC'] = lae.transform(test_csv['SCC'])

lae.fit(train_csv['CALC'])
train_csv['CALC'] = lae.transform(train_csv['CALC'])
test_csv['CALC'] = lae.transform(test_csv['CALC'])

lae.fit(train_csv['MTRANS'])
train_csv['MTRANS'] = lae.transform(train_csv['MTRANS'])
test_csv['MTRANS'] = lae.transform(test_csv['MTRANS'])

# print(train_csv['MTRANS'])
# # print(train_csv['CALC'])
# print(train_csv['SCC'])
# print(train_csv['CAEC'])
# print(train_csv['SMOKE'])



X = train_csv.drop(['NObeyesdad','SMOKE'], axis=1)
y = train_csv['NObeyesdad']
test_csv = test_csv.drop(['SMOKE'], axis=1)

# y = y.values.reshape(-1, 1)
# ohe = OneHotEncoder(sparse=False,
#                     # handle_unknown= 'infrequent_if_exist'
#                     )
# ohe.fit(y)
# y1 = ohe.transform(y)

# print(y1)
# print(y1.shape)  # (20758, 7)

n_splits= 5
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=948)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=948, stratify=y)

# mms1 = ['Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CH2O', 'TUE', 'FAF']
# mms = MinMaxScaler()
# X_train[mms1] = mms.fit_transform(X_train[mms1])
# X_test[mms1] = mms.fit_transform(X_test[mms1])
# test_csv[mms1] = mms.fit_transform(test_csv[mms1])

# rbs1 = ['Age', 'Height', 'Weight']
# rbs = RobustScaler(quantile_range=(5,95))
# X_train[rbs1] = rbs.fit_transform(X_train[rbs1])
# X_test[rbs1] = rbs.fit_transform(X_test[rbs1])
# test_csv[rbs1] = rbs.fit_transform(test_csv[rbs1])



# model = Sequential()
# model.add(Dense(19, input_shape= (16, ),activation='relu'))
# model.add(Dense(97,activation='relu'))
# model.add(Dense(9,activation='relu'))
# model.add(Dense(21,activation='relu'))
# model.add(Dense(16,activation='relu'))
# model.add(Dense(21,activation='relu'))
# model.add(Dense(7, activation='softmax'))

# model = RandomForestClassifier(random_state=713, n_estimators=420, verbose=1, )
# model = XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=4)




parameters = [
    {'n_estimators': [100,200], 'max_depth': [6,12,18],
     'min_samples_leaf' : [3, 10]},
    {'max_depth' : [6, 8, 10, 12], 'min_samples_leaf' : [3, 5, 7, 10]},
    {'min_samples_leaf' : [3, 5, 7, 10],
     'min_samples_split' : [2, 3, 5, 10]},
    {'min_samples_split' : [2, 3, 5,10]},
    {'n_jobs' : [-1, 10, 20], 'min_samples_split' : [2, 3, 5, 10]}   
]

 #2. 모델 구성
model = GridSearchCV(RandomForestClassifier(), parameters, cv=kfold, verbose=1,
                    #  random_state = 3,
                    # refit = True,     # default
                     n_jobs=-1
                     )



start_time = time.time()
model.fit(X_train, y_train)
end_time = time.time()
print("최적의 매개변수 : ", model.best_estimator_)
# 최적의 매개변수 :  SVC(C=1, kernel='linear')

print("최적의 파라미터 : ", model.best_params_)
# 최적의 파라미터 :  {'C': 1, 'degree': 3, 'kernel': 'linear'}

print('best_score : ', model.best_score_)
print('model.score : ', model.score(X_test, y_test))
# results = model.score(X_test, y_test)
# print(results)
y_predict = model.predict(X_test)
acc = accuracy_score(y_test, y_predict)
print("accuracy_score : ", acc)

y_pred_best = model.best_estimator_.predict(X_test)
print("최적튠 ACC : " , accuracy_score(y_test, y_pred_best))
# best_score :  0.975 
# model.score :  0.9333333333333333
print("걸린시간 : ", round(end_time - start_time, 2), "초")

# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics='acc')
# es = EarlyStopping(monitor='acc', mode='max', patience=500, verbose=20, restore_best_weights=True)
# model.fit(X_train, y_train, epochs=100, batch_size=500, validation_split=0.2, callbacks=[es], verbose=2)
# model.fit(X_train, y_train)
# results = model.score(X_test, y_test)

# # results = model.evaluate(X_test, y_test)

# y_predict = model.predict(X_test) 
# y_test = ohe.inverse_transform(y_test)
# y_predict = ohe.inverse_transform(y_predict)


y_submit = model.predict(test_csv)  
# y_submit = ohe.inverse_transform(y_submit)

y_submit = pd.DataFrame(y_submit)
submission_csv['NObeyesdad'] = y_submit
# print(y_submit)
# print("ACC : ", results)

# fs = f1_score(y_test, y_predict, average='macro')
# print("f1_score : ", fs)
    
# submission_csv.to_csv(path + "submisson_02_08_2_random_forest.csv", index=False)
submission_csv.to_csv(path + "submisson_02_10_5_rf.csv", index=False)


# best_score :  0.8982092015043479
# model.score :  0.9075144508670521
# accuracy_score :  0.9075144508670521
# 최적튠 ACC :  0.9075144508670521
# 걸린시간 :  19.54 초

#random
# f1_score :  0.8469567014233303



# xgb
# f1_score :  0.8799252701582044


# best_score :  0.8979257387284753
# model.score :  0.9071933204881182
# accuracy_score :  0.9071933204881182
# 최적튠 ACC :  0.9071933204881182
# 걸린시간 :  19.75 초

# best_score :  0.8987759824814029
# model.score :  0.9071933204881182
# accuracy_score :  0.9071933204881182
# 최적튠 ACC :  0.9071933204881182
# 걸린시간 :  308.94 초

















