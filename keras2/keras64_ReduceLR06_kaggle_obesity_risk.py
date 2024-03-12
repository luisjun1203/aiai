import numpy as np
import pandas as pd
import time
from keras.models import Sequential, Model, load_model
from keras. layers import Dense, Conv1D, SimpleRNN, LSTM, Flatten, GRU, Dropout, Input, concatenate
from keras. callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
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
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.decomposition import PCA 


path = "c:\\_data\\kaggle\\obesity_risk\\"


train_csv = pd.read_csv(path + "train.csv", index_col=0)
test_csv = pd.read_csv(path + "test.csv", index_col=0)
submission_csv = pd.read_csv(path + "sample_submission.csv")



# print(train_csv.shape)      # (20758, 17)
# print(test_csv.shape)       # (13840, 16)
# print(submission_csv.shape) # (13840, 1)



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





X = train_csv.drop(['NObeyesdad','SMOKE'], axis=1)
y = train_csv['NObeyesdad']
test_csv = test_csv.drop(['SMOKE'], axis=1)

y = y.values.reshape(-1, 1)
ohe = OneHotEncoder(sparse=False,
                    # handle_unknown= 'infrequent_if_exist'
                    )
ohe.fit(y)
y1 = ohe.transform(y)

# print(y1)
# print(y1.shape)  # (20758, 7)





X_train, X_test, y_train, y_test = train_test_split(X, y1, test_size=0.15, shuffle=True, random_state=43297347, stratify=y)

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






#2. 모델 구성



from keras.optimizers import Adam
learning_rate = 0.01
epochs = 500

model = Sequential()
model.add(Dense(19, input_shape= (15, ),activation='relu'))
model.add(Dense(97,activation='relu'))
model.add(Dense(9,activation='relu'))
model.add(Dense(21,activation='relu'))
model.add(Dense(16,activation='relu'))
model.add(Dense(21,activation='relu'))
model.add(Dense(7, activation='softmax'))


import datetime
date = datetime.datetime.now()
print(date)                     # 2024-01-17 10:52:59.857389
date = date.strftime("%m%d_%H%M")                   # _는 str      
print(date)                     # 0117_1058
print(type(date))               # <class 'str'>

path = "..\\_data\\_save\\MCP\\"
filename = '{epoch:05d}-{val_loss:.4f}-{loss:.4f}.hdf5'            # 04d : 4자리 정수표현, 4f : 소수4번째자리까지 표현, 예) 1000_0.3333.hdf5
filepath = "".join([path, 'k64_kaggle_obsity_risk_',date,'_', filename])
mcp = ModelCheckpoint(monitor='val_loss', mode='min', verbose=1, save_best_only=True, filepath=filepath)



start_time = time.time()
rlr = ReduceLROnPlateau(monitor='val_loss', patience=10, mode='accuracy', verbose=1, factor=0.5)

model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=learning_rate), metrics='acc')
es = EarlyStopping(monitor='val_loss', mode='min', patience=50, verbose=1, restore_best_weights=True)
model.fit(X_train, y_train, epochs=epochs, batch_size=500, validation_split=0.2, callbacks=[es, mcp, rlr], verbose=2)

results = model.evaluate(X_test, y_test)
end_time = time.time()

y_predict = model.predict(X_test) 
y_test = ohe.inverse_transform(y_test)
y_predict = ohe.inverse_transform(y_predict)


acc = accuracy_score(y_test, y_predict)

print("lr : {0}, epochs : {1}, ACC : {2}, 로스 : {3}, 걸린시간 : {4}초 ".format(learning_rate, epochs, acc, results, round(end_time - start_time, 2) ))

# lr : 1.0, epochs : 300, ACC : 0.19492614001284522, 로스 : [1.9399042129516602, 0.19492614269256592], 걸린시간 : 4.35초 
# lr : 0.1, epochs : 300, ACC : 0.19492614001284522, 로스 : [1.9328259229660034, 0.19492614269256592], 걸린시간 : 3.96초
# lr : 0.01, epochs : 300, ACC : 0.8484264611432242, 로스 : [0.4030408263206482, 0.8484264612197876], 걸린시간 : 10.62초
# lr : 0.001, epochs : 300, ACC : 0.8336544637122671, 로스 : [0.42726391553878784, 0.8336544632911682], 걸린시간 : 17.48초
# lr : 0.0001, epochs : 300, ACC : 0.7906229929351317, 로스 : [0.5101497173309326, 0.7906230092048645], 걸린시간 : 17.72초



# rlr 적용
# lr : 0.01, epochs : 500, ACC : 0.861271676300578, 로스 : [0.37194114923477173, 0.8612716794013977], 걸린시간 : 17.13초











