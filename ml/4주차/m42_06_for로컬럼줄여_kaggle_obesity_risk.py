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



parameters = {
    'n_estimators': 4000,  # 부스팅 라운드의 수/ 디폴트 100/ 1 ~ inf/ 정수
    'learning_rate': 0.05,  # 학습률/ 디폴트 0.3/0~1/
    'max_depth': 8,  # 트리의 최대 깊이/ 디폴트 6/ 0 ~ inf/ 정수
    'min_child_weight': 1,  # 자식에 필요한 모든 관측치에 대한 가중치 합의 최소/ 디폴트 1 / 0~inf
    'gamma': 0.1,  # 리프 노드를 추가적으로 나눌지 결정하기 위한 최소 손실 감소/ 디폴트 0/ 0~ inf
    'subsample': 0.6,  # 각 트리마다의 관측 데이터 샘플링 비율/ 디폴트 1 / 0~1
    'colsample_bytree': 0.6,  # 각 트리 구성에 필요한 컬럼(특성) 샘플링 비율/ 디폴트 1 /9 0~1
    'colsample_bylevel': 0.6, #  디폴트 1 / 0~1
    'colsample_bynode': 0.6, #  디폴트 1 / 0~1
    'reg_alpha' : 0.5,   # 디폴트 0 / 0 ~ inf / L1 절대값 가중치 규제(제한) / alpha
    'reg_lambda' :   0.7,   # 디폴트 1 / 0 ~ inf / L2 제곱 가중치 규제(제한) / lambda
    
}
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=777, test_size=0.2)


# 2. 모델 구성
importance = [] #  모델이 특정 특성 수로 훈련되었을 때의 정확도(accuracy)를 포함

for i in range(X.shape[1], 0, -1):  # X.shape[1] : 열의수(특성), (시작, 끝, step)


    model = XGBClassifier()
    model.set_params(early_stopping_rounds = 10, **parameters)



# 3. 훈련
    model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_test, y_test)],
            verbose=1,
            eval_metric ='mlogloss' 
          )


# 4. 평가, 예측


    y_predict = model.predict(X_test)
    acc = accuracy_score(y_test, y_predict)
    
    importance.append((i, acc))
    
    

    if i > 1:  
        
        fI = model.feature_importances_
        least_important = np.argmin(fI)     # fI 배열 내에서 가장 낮은 중요도 값을 가진 특성의 인덱스를 찾는다.
        X_train = np.delete(X_train, least_important, axis=1)
        X_test = np.delete(X_test, least_important, axis=1)
   
for n_features, acc in importance:
    print(f"특성: {n_features}, Accuracy: {acc:.4f}")   

best_feature_count, best_accuracy = max(importance, key=lambda x: x[1]) # max함수가 importance 리스트 X에 대해 X[1]값인 acc를 반환

print(f"가장 좋았던 정확도: {best_accuracy:.4f}, 해당할 때의 특성 수: {best_feature_count}") 


# 특성: 19, Accuracy: 0.9092
# 특성: 18, Accuracy: 0.9094
# 특성: 17, Accuracy: 0.9087
# 특성: 16, Accuracy: 0.9075
# 특성: 15, Accuracy: 0.9078
# 특성: 14, Accuracy: 0.9080
# 특성: 13, Accuracy: 0.9008
# 특성: 12, Accuracy: 0.8993
# 특성: 11, Accuracy: 0.8943
# 특성: 10, Accuracy: 0.8938
# 특성: 9, Accuracy: 0.8909
# 특성: 8, Accuracy: 0.8890
# 특성: 7, Accuracy: 0.8895
# 특성: 6, Accuracy: 0.8817
# 특성: 5, Accuracy: 0.8752
# 특성: 4, Accuracy: 0.8764
# 특성: 3, Accuracy: 0.8350
# 특성: 2, Accuracy: 0.4391
# 특성: 1, Accuracy: 0.3485
# 가장 좋았던 정확도: 0.9094, 해당할 때의 특성 수: 18