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
from sklearn.model_selection import  HalvingGridSearchCV,HalvingRandomSearchCV
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold,cross_val_predict
from lightgbm import LGBMClassifier

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




X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, shuffle=True, random_state=86306949)

splits = 3
kfold = StratifiedKFold(n_splits=splits, shuffle=True, random_state=95969402)

parameters = {
     'LG__num_leaves': [31],  # 트리가 가질 수 있는 최대 잎의 수
    'LG__max_depth': [ 0],  # 트리의 최대 깊이
    'LG__learning_rate': [0.05],  # 학습률
    'LG__subsample': [0.8],  # 각 트리를 구축할 때 사용하는 데이터의 비율
    'LG__verbosity': [2],  # LightGBM의 실행 중 정보 출력 설정
    'LG__min_child_samples': [20],  # 리프 노드가 되기 위해 필요한 최소 샘플 수
    'LG__min_child_weight': [0.001],  # 자식에 필요한 모든 관측치에 대한 가중치 합의 최소값
    'LG__colsample_bytree': [0.6],  # 각 트리를 구축할 때 사용하는 특성의 비율
    'LG__reg_alpha': [0.1],  # L1 규제 항
    'LG__reg_lambda': [0.1],  # L2 규제 항
    'LG__n_estimators': [200],  # 부스팅 단계의 수
    'LG__boosting_type': ['gbdt']  # 부스팅 타입
      
}

# parameters = {
#         'LG__num_leaves': [31],
#         'LG__max_depth': [-1],
#         'LG__learning_rate': [0.05],
#         'LG__n_estimators': [100],
#         'LG__min_child_samples': [50],
#         'LG__subsample': [0.8],  
#     }


pipe = Pipeline([('MM', MinMaxScaler()),
                 ('LG', LGBMClassifier(random_state=2560064, objective='multiclass'))])

model = HalvingGridSearchCV(pipe, param_grid=parameters,
                     cv = kfold,
                     verbose=1,
                     refit=True,
                     n_jobs=-1,   
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
submission_csv.to_csv(path + "submisson_02_16_1_lgbm.csv", index=False)






# random_state :  86306949
# random_state :  95969402
# random_state :  2560064



