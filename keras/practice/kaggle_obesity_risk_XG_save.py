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

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, shuffle=True, random_state=698423134,stratify=y)

splits = 5
kfold = StratifiedKFold(n_splits=splits, shuffle=True, random_state=2928297790)

parameters = {
    'XGB__n_estimators': [30, 50 ,300],  # 부스팅 라운드의 수
    'XGB__learning_rate': [0.05, 0.1],  # 학습률
    'XGB__max_depth': [3, 6, 9],  # 트리의 최대 깊이
    'XGB__min_child_weight': [1, 5, 10],  # 자식에 필요한 모든 관측치에 대한 가중치 합의 최소
    'XGB__gamma': [0.5, 1, 1.5, 2],  # 리프 노드를 추가적으로 나눌지 결정하기 위한 최소 손실 감소
    'XGB__subsample': [0.6, 0.8, 1.0],  # 각 트리마다의 관측 데이터 샘플링 비율
    'XGB__colsample_bytree': [0.6, 0.8, 1.0],  # 각 트리 구성에 필요한 컬럼(특성) 샘플링 비율
    'XGB__objective': ['multi:softmax'],  # 학습 태스크 파라미터
    'XGB__num_class': [20],  # 분류해야 할 전체 클래스 수, 멀티클래스 분류인 경우 설정
    'XGB__verbosity' : [1] 
}

# parameters = {
#     'XGB__n_estimators': [300],  # 부스팅 라운드의 수
#     'XGB__learning_rate': [0.05],  # 학습률
#     'XGB__max_depth': [9],  # 트리의 최대 깊이
#     'XGB__min_child_weight': [5, 10],  # 자식에 필요한 모든 관측치에 대한 가중치 합의 최소
#     'XGB__gamma': [1, 1.5],  # 리프 노드를 추가적으로 나눌지 결정하기 위한 최소 손실 감소
#     'XGB__subsample': [0.6, 0.8],  # 각 트리마다의 관측 데이터 샘플링 비율
#     'XGB__colsample_bytree': [0.6],  # 각 트리 구성에 필요한 컬럼(특성) 샘플링 비율
#     'XGB__objective': ['multi:softmax'],  # 학습 태스크 파라미터
#     'XGB__num_class': [16],  # 분류해야 할 전체 클래스 수, 멀티클래스 분류인 경우 설정
#     'XGB__verbosity' : [1] 
# }





pipe = Pipeline([('SS',StandardScaler()),
                 ('XGB', XGBClassifier(random_state=3608501786))])

model = HalvingGridSearchCV(pipe, parameters,
                     cv = kfold,
                     verbose=1,
                     refit=True,
                     n_jobs=-1,   
                    # n_iter=10 # 디폴트 10
                    factor=2,
                    min_resources=40)


model.fit(X_train,y_train)



print("최적의 매개변수:",model.best_estimator_)
print("최적의 파라미터:",model.best_params_)
print("best_score:",model.best_score_) 
print("model.score:", model.score(X_test,y_test)) 

y_predict=model.predict(X_test)
print("acc.score:", accuracy_score(y_test,y_predict))
y_pred_best=model.best_estimator_.predict(X_test)

print("best_acc.score:",accuracy_score(y_test,y_pred_best))

y_submit = model.predict(test_csv)  
y_submit = lae.inverse_transform(y_submit)
y_submit = pd.DataFrame(y_submit)
# y_submit = lae.inverse_transform(y_submit)
submission_csv['NObeyesdad'] = y_submit
print(y_submit)

submission_csv.to_csv(path + "submisson_02_19_88_xgb.csv", index=False)


import pickle
path = "c://_data//_save//_pickle_test//kaggle_obesity_risk\\"
pickle.dump(model, open(path + "kaggle_obesity_risk_save.dat", 'wb'))



# best_score: 0.895682805794959
# model.score: 0.9325899645210339
# acc.score: 0.9325899645210339
# best_acc.score: 0.9325899645210339

#Maxabs X
#Robust X


# submisson_02_16_2_xgb.csv
# 최적의 파라미터: {'XGB__verbosity': 1, 'XGB__subsample': 0.6, 'XGB__objective': 'multi:softmax',
#            'XGB__num_class': 16, 'XGB__n_estimators': 200, 'XGB__min_child_weight': 1,
#            'XGB__max_depth': 9, 'XGB__learning_rate': 0.05, 'XGB__gamma': 0.5, 'XGB__colsample_bytree': 0.6}
# best_score: 0.9010283087624776
# model.score: 0.9233230571612074
# acc.score: 0.9233230571612074
# best_acc.score: 0.9233230571612074


##### random_state :  698423134
##### random_state :  2928297790
##### random_state :  3608501786


# random_state :  2726041495
# random_state :  4125478883
# random_state :  3856098980


# 최적의 파라미터: {'XGB__colsample_bytree': 0.8, 'XGB__gamma': 1, 'XGB__learning_rate': 0.1, 'XGB__max_depth': 6, 'XGB__min_child_weight': 1, 'XGB__n_estimators': 300, 'XGB__num_class': 16, 'XGB__objective': 'multi:softmax', 'XGB__subsample': 0.6, 'XGB__verbosity': 1}
# best_score: 0.9044786705159397
# model.score: 0.9251766217084136
# acc.score: 0.9251766217084136
# best_acc.score: 0.9251766217084136



# 최적의 파라미터: {'XGB__colsample_bytree': 0.6, 'XGB__gamma': 1, 'XGB__learning_rate': 0.05, 'XGB__max_depth': 9, 'XGB__min_child_weight': 1, 'XGB__n_estimators': 400, 'XGB__num_class': 17, 'XGB__objective': 'multi:softmax', 'XGB__subsample': 0.6, 'XGB__verbosity': 1}
# best_score: 0.9050869974352823
# model.score: 0.926461143224149
# acc.score: 0.926461143224149
# best_acc.score: 0.926461143224149


# 3439645700#########!!!!!!!


# best_score: 0.8963765992772732
# model.score: 0.9330443159922929
# acc.score: 0.9330443159922929
# best_acc.score: 0.9330443159922929