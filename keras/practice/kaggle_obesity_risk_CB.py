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
from catboost import CatBoostClassifier
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import  HalvingGridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold,cross_val_predict


path = "c:\\_data\\kaggle\\obesity_risk\\"


train_csv = pd.read_csv(path + "train.csv", index_col=0)
test_csv = pd.read_csv(path + "test.csv", index_col=0)
submission_csv = pd.read_csv(path + "sample_submission.csv")


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


# print(y)

# y = lae.fit_transform(y)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, shuffle=True, random_state=43297347, stratify=y)


splits = 3
kfold = StratifiedKFold(n_splits=splits, shuffle=True, random_state=28)

parameters = {
    'CB__iterations': [100, 500, 1000],  # 트리의 개수
    'CB__learning_rate': [0.01, 0.1, 0.5],  # 학습률
    'CB__depth': [3, 6, 10],  # 트리의 깊이
    'CB__l2_leaf_reg': [1, 3, 5],  # L2 정규화 계수
    'CB__border_count': [32, 64, 128],  # 분할을 위한 피처 값의 개수
    'CB__loss_function': ['MultiClass'],  # 손실 함수
    'CB__eval_metric': ['Accuracy'],  # 평가 지표
    'CB__random_seed': [42]
}



# model = CatBoostClassifier(random_seed=3)

pipe = Pipeline([('MM', MinMaxScaler()),
                 ('CB', CatBoostClassifier(random_seed=3, verbose=False))])

model = HalvingGridSearchCV(pipe, param_grid=parameters,
                     cv = kfold,
                     verbose=1,
                     refit=True,
                     n_jobs=3,   
                    # n_iter=10 # 디폴트 10
                    factor=2,
                    min_resources=462)


model.fit(X_train,y_train)


print("최적의 매개변수:",model.best_estimator_)
print("최적의 파라미터:",model.best_params_)
print("best_score:",model.best_score_) 
print("model.score:", model.score(X_test,y_test)) 

y_predict=model.predict(X_test)
print("acc.score:", accuracy_score(y_test,y_predict))
y_pred_best=model.best_estimator_.predict(X_test)

print("best_acc.score:",accuracy_score(y_test,y_pred_best))



# model.fit(X_train, y_train)
# results = model.score(X_test, y_test)
# y_predict = model.predict(X_test)
# acc = accuracy_score(y_test, y_predict)
# print("acc : ", acc)

y_submit = model.predict(test_csv)  
y_submit = pd.DataFrame(y_submit)
submission_csv['NObeyesdad'] = y_submit
# print(y_submit)
# print("ACC : ", results)

# fs = f1_score(y_test, y_predict, average='macro')
# print("f1_score : ", fs)
    
print(y_submit)
submission_csv.to_csv(path + "submisson_02_15_1_cb.csv", index=False)


# best_score: 0.9027198614929485
# model.score: 0.9155427103403982
# acc.score: 0.9155427103403982
# best_acc.score: 0.9155427103403982