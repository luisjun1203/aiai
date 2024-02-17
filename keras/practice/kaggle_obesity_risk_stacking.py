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
from sklearn.ensemble import RandomForestClassifier ,VotingClassifier, StackingClassifier
from xgboost import XGBClassifier
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import  HalvingGridSearchCV,HalvingRandomSearchCV
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
import xgboost as xgb
from lightgbm import LGBMClassifier
import lightgbm as lgb 
from catboost import CatBoostClassifier
from sklearn.tree import DecisionTreeClassifier


path = "c:\\_data\\kaggle\\obesity_risk\\"


train_csv = pd.read_csv(path + "train.csv", index_col=0)
test_csv = pd.read_csv(path + "test.csv", index_col=0)
submission_csv = pd.read_csv(path + "sample_submission.csv")

#################### 이상치 하나 변경 ##################################

test_csv.loc[test_csv['CALC']=='Always', 'CALC'] = 'Frequently'

##################### 교통수단 컬럼 살짝 변경 #######################################
train_csv.loc[train_csv['MTRANS']=='Bike', 'MTRANS'] = 'Public_Transportation'
train_csv.loc[train_csv['MTRANS']=='Motorbike', 'MTRANS'] = 'Automobile'

test_csv.loc[test_csv['MTRANS']=='Bike', 'MTRANS'] = 'Public_Transportation'
test_csv.loc[test_csv['MTRANS']=='Motorbike', 'MTRANS'] = 'Automobile'

# print(np.unique(train_csv['MTRANS'], return_counts=True))
# print(np.unique(test_csv['MTRANS'], return_counts=True))


################# 운동량 컬럼 추가 ###################################################################

train_csv['Exercise_Score'] = train_csv['FAF'] - train_csv['TUE'] + train_csv['FCVC']
test_csv['Exercise_Score'] = test_csv['FAF'] - test_csv['TUE'] + test_csv['FCVC']

# print(train_csv['Exercise_Score'])
#################### 식습관 가족력 컬럼 추가 ##############################################

def classify_diet(caec, calc, favc, family_history):
    if family_history == 'yes':
        return 'Moderate'
    elif caec == 'Always' and calc == 'Frequently' and favc == 'yes':
        return 'Unhealthy'
    elif caec == 'Frequently' and calc == 'Always' and favc == 'yes':
        return 'Unhealthy'
    elif caec == 'Sometimes' and calc == 'Frequently'and favc == 'yes':
        return 'Moderate'
    elif caec == 'Sometimes' and calc == 'Always'and favc == 'yes':
        return 'Moderate'
    else:
        return 'Healthy'
    
train_csv['Diet_Class'] = train_csv.apply(lambda row: classify_diet(row['CAEC'], row['CALC'], row['FAVC'], row['family_history_with_overweight']), axis=1)  
test_csv['Diet_Class'] = test_csv.apply(lambda row: classify_diet(row['CAEC'], row['CALC'], row['FAVC'], row['family_history_with_overweight']), axis=1)  

# print(np.unique(train_csv['Diet_Class'], return_counts=True))
# print(np.unique(train_csv['family_history_with_overweight'], return_counts=True))
# print(train_csv['Diet_Class'])
# print(test_csv['Diet_Class'])



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


lae.fit(train_csv['Diet_Class'])
train_csv['Diet_Class'] = lae.transform(train_csv['Diet_Class'])
test_csv['Diet_Class'] = lae.transform(test_csv['Diet_Class'])




# print(train_csv['MTRANS'])
# # print(train_csv['CALC'])
# print(train_csv['SCC'])
# print(train_csv['CAEC'])
# print(train_csv['SMOKE'])

train_csv['BMI'] = 1.3 * (train_csv['Weight'] / (train_csv['Height']*2.5))            # 이상치에 좋은 bmi계산법
test_csv['BMI'] = 1.3 * (test_csv['Weight'] / (test_csv['Height']*2.5))


# train_csv['BMI'] = (train_csv['Weight'] / (train_csv['Height'] * train_csv['Height']))
# test_csv['BMI'] = (test_csv['Weight'] / (test_csv['Height'] * test_csv['Height']))



X = train_csv.drop(['NObeyesdad'], axis=1)
y = train_csv['NObeyesdad']


# lae.fit(y)
# y = lae.transform(y)
# def auto(a):
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, shuffle=True, random_state=3439645700, stratify=y)

# splits = 3
# kfold = StratifiedKFold(n_splits=splits, shuffle=True, random_state=4125478883)
base_models = [
    ('xgb', XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=9, gamma=0.5, colsample_bytree=0.6)),
    ('lgb', LGBMClassifier(colsample_bytree=0.4097712934687264,
               lambda_l1=0.009667446568254372, lambda_l2=0.040186414373018,
               learning_rate=0.03096221154683276, max_depth=10,
               metric='multi_logloss', min_child_samples=26, n_estimators=500,
               num_class=7, objective='multiclass', random_state=42,
               subsample=0.9535797422450176, verbosity=-1)),
    ('cat', CatBoostClassifier(iterations=300, learning_rate=0.05, depth=10))
]

meta_model = RandomForestClassifier(n_estimators=100)

stacking_model = StackingClassifier(estimators=base_models, final_estimator=meta_model, cv=5)

stacking_model.fit(X_train, y_train)


y_predict = stacking_model.predict(X_test)

stacking_accuracy = accuracy_score(y_test, y_predict)
y_submit = stacking_model.predict(test_csv)
# y_submit = lae.inverse_transform(y_submit)


y_submit = pd.DataFrame(y_submit)
submission_csv['NObeyesdad'] = y_submit
print(y_submit)
print("Stacking Model Accuracy:", stacking_accuracy) 


submission_csv.to_csv(path + "submisson_02_17_4_stacking.csv", index=False)


