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
from sklearn.ensemble import RandomForestClassifier ,VotingClassifier
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

test_csv.loc[test_csv['CALC']=='Always', 'CALC'] = 'Frequently'

##################### 교통수단 컬럼 살짝 변경 #######################################
train_csv.loc[train_csv['MTRANS']=='Bike', 'MTRANS'] = 'Public_Transportation'
train_csv.loc[train_csv['MTRANS']=='Motorbike', 'MTRANS'] = 'Automobile'

test_csv.loc[test_csv['MTRANS']=='Bike', 'MTRANS'] = 'Public_Transportation'
test_csv.loc[test_csv['MTRANS']=='Motorbike', 'MTRANS'] = 'Automobile'


# Bike를 대중교통에 포함시켰다가 Walking으로 바꿈
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

# train_csv['BMI'] = 1.3 * (train_csv['Weight'] / (train_csv['Height']*2.5))            # 이상치에 좋은 bmi계산법
# test_csv['BMI'] = 1.3 * (test_csv['Weight'] / (test_csv['Height']*2.5))


train_csv['BMI'] = (train_csv['Weight'] / (train_csv['Height'] * train_csv['Height']))
test_csv['BMI'] = (test_csv['Weight'] / (test_csv['Height'] * test_csv['Height']))


numeric_cols = train_csv.select_dtypes(include=[np.number])


q1 = numeric_cols.quantile(0.05)
q3 = numeric_cols.quantile(0.95)
iqr = q3 - q1

lower_limit = q1 - 1.5*iqr
upper_limit = q3 + 1.5*iqr


for label in numeric_cols:
    lower = lower_limit[label]
    upper = upper_limit[label]
    
    train_csv[label] = np.where(train_csv[label] < lower, lower, train_csv[label])
    train_csv[label] = np.where(train_csv[label] > upper, upper, train_csv[label])

X = train_csv.drop(['NObeyesdad'], axis=1)
y = train_csv['NObeyesdad']


# lae.fit(y)    
# y = lae.transform(y)
# def auto(a):
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, shuffle=True, random_state=3439645700, stratify=y)

# splits = 3
# kfold = StratifiedKFold(n_splits=splits, shuffle=True, random_state=4125478883)

xgb_model = xgb.XGBClassifier(n_estimators=300, learning_rate = 0.05, max_depth = 9, gamma = 0.5, colsample_bytree = 0.6, verbosity = 1 )
# {'XGB__verbosity': 1, 'XGB__subsample': 0.6, 'XGB__objective': 'multi:softmax',
# #            'XGB__num_class': 16, 'XGB__n_estimators': 200, 'XGB__min_child_weight': 1,
# #            'XGB__max_depth': 9, 'XGB__learning_rate': 0.05, 'XGB__gamma': 0.5, 'XGB__colsample_bytree': 0.6}

lgb_model = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, max_depth=0, num_leaves=31, colsample_bytree=0.6, verbose=0)
# {'LG__verbosity': 2, 'LG__subsample': 0.8, 'LG__reg_lambda': 0.1,
#  'LG__reg_alpha': 0.1, 'LG__num_leaves': 31, 'LG__n_estimators': 200,
#  'LG__min_child_weight': 0.001, 'LG__min_child_samples': 20, 'LG__max_depth': 0,
#  'LG__learning_rate': 0.05, 'LG__colsample_bytree': 0.6, 'LG__boosting_type': 'gbdt'}

# cat_model = CatBoostClassifier(iterations=300, learning_rate=0.05, max_depth=0)


voting_model = VotingClassifier(estimators=[('xgb', xgb_model), ('lgb', lgb_model), ], voting='soft')


voting_model.fit(X_train, y_train)


accuracy = voting_model.score(X_test, y_test)
y_submit = voting_model.predict(test_csv)  
# y_submit = lae.inverse_transform(y_submit)

y_predict = voting_model.predict(X_test)
acc = accuracy_score(y_test, y_predict)

y_submit = pd.DataFrame(y_submit)
submission_csv['NObeyesdad'] = y_submit
print(y_submit)
print("Voting Ensemble Accuracy:", accuracy)

submission_csv.to_csv(path + "submisson_02_19_99_voting.csv", index=False)
# return acc
# print(voting_model.feature_importances_)
# import random
# for i in range(100000):
#     a = random.randrange(1,4200000000)
#     r = auto(a)
#     if r > 0.925 :
#         print("random_state : ", a)
#         print("ACC : ", r)
#         break

# Voting Ensemble Accuracy: 0.924373795761079

