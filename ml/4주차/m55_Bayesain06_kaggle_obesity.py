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
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import  HalvingGridSearchCV,HalvingRandomSearchCV
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
from imblearn.over_sampling import SMOTE
from bayes_opt import BayesianOptimization

path = "c:\\_data\\kaggle\\obesity_risk\\"


train_csv = pd.read_csv(path + "train.csv", index_col=0)
test_csv = pd.read_csv(path + "test.csv", index_col=0)
submission_csv = pd.read_csv(path + "sample_submission.csv")

test_csv.loc[test_csv['CALC']=='Always', 'CALC'] = 'Sometimes'

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

# BMI 컬럼추가
train_csv['BMI'] = 1.3 * (train_csv['Weight'] / (train_csv['Height']*2.5))
test_csv['BMI'] = 1.3 * (test_csv['Weight'] / (test_csv['Height']*2.5))


# print(train_csv.info())
# print(test_csv.info())



X = train_csv.drop(['NObeyesdad'], axis=1)
y = train_csv['NObeyesdad']
# test_csv = test_csv.drop(['SMOKE'], axis=1)

lae.fit(y)
y = lae.transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, shuffle=True, random_state=698423134, stratify=y)


bayesian_params = {
    'learning_rate' : (0.001, 1),
    'max_depth' : (3, 10),
    'num_leaves' : (24, 40),
    'min_child_samples' : (10, 200),
    'min_child_weight' : (1, 50),
    'subsample' : (0.5, 1),
    'colsample_bytree' : (0.5, 1),
    'max_bin' : (9, 500),
    'reg_lambda' : (-0.001, 10),
    'reg_alpha' : (0.01, 50),
}


def xgb_hamsu(learning_rate, max_depth, num_leaves, min_child_samples, min_child_weight,
              subsample, colsample_bytree, max_bin, reg_alpha, reg_lambda):

    params = {
        'n_estimators' : 100, 
        'learning_rate' : learning_rate,
        'max_depth' : int(round(max_depth)),                    # 무조건 정수형
        'num_leaves' : int(round(num_leaves)),
        'min_child_samples' : int(round(min_child_samples)),
        'min_child_weight' : int(round(min_child_weight)),
        'subsample' : max(min(subsample, 1), 0),                # 0~1 사이의 값
        'colsample_bytree' : colsample_bytree,
        'max_bin' : max(int(round(max_bin)), 10),               # 무조건 10이상
        'reg_lambda' : max(reg_lambda, 0),                      # 무조건 양수만
        'reg_alpha' : reg_alpha,  
}

    model = XGBClassifier(**params, n_jobs = -1)

    model.fit(X_train, y_train,
              eval_set=[(X_train, y_train), (X_test, y_test)],
              eval_metric = 'mlogloss',
              verbose = 0,
              early_stopping_rounds = 50
              )

    y_predict = model.predict(X_test)
    results = accuracy_score(y_test, y_predict)
    return results

bay = BayesianOptimization(
    f = xgb_hamsu,
    pbounds=bayesian_params,
    random_state=3
) 
n_iter = 100
start = time.time()
bay.maximize(init_points=5,
             n_iter=n_iter)
end = time.time()

print(bay.max)
print(n_iter, " 번 걸린시간 : ", round(end - start, 2), "초" )

# {'target': 0.9335260115606936, 'params': {'colsample_bytree': 0.5322019907107679,
# 'learning_rate': 0.32216761371904284, 'max_bin': 286.43081908617944,
# 'max_depth': 9.491195294425072, 'min_child_samples': 59.58659787483409, 
# 'min_child_weight': 3.115382420131752, 'num_leaves': 38.26592119766204,
# 'reg_alpha': 6.788313120836739, 'reg_lambda': 6.866529498320359, 'subsample': 0.629907986737716}}
# 100  번 걸린시간 :  56.35 초

    
    