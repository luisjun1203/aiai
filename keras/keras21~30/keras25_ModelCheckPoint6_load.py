# restore_best_weights
# save_best_only
# 에 대한 고찰


# keras09_1_boston.py
import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import r2_score
from sklearn.datasets import load_boston
from keras.models import Sequential, load_model
from keras.layers import Dense
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, StandardScaler, RobustScaler
from keras.callbacks import EarlyStopping, ModelCheckpoint 


datasets = load_boston()

X = datasets.data
y = datasets.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, shuffle=True, random_state=20)

mms = MinMaxScaler()
mms.fit(X_train)
X_train = mms.transform(X_train)
X_test = mms.transform(X_test)

# # # 2. 모델 구성
# model = Sequential()
# model.add(Dense(19,input_dim=13,activation='relu'))
# model.add(Dense(97,activation='relu'))
# model.add(Dense(9,activation='relu'))
# model.add(Dense(21,activation='relu'))
# model.add(Dense(28,activation='relu'))
# model.add(Dense(1))
# # model.summary()

# import datetime
# date = datetime.datetime.now()
# print(date)                     # 2024-01-17 10:52:59.857389
# date = date.strftime("%m%d_%H%M")                   # _는 str      
# print(date)                     # 0117_1058
# print(type(date))               # <class 'str'>

# path = "..\\_data\\_save\\MCP\\"
# filename = '{epoch:04d}-{val_loss:.4f}.hdf5'            # 04d : 4자리 정수표현, 4f : 소수4번째자리까지 표현, 예) 1000_0.3333.hdf5
# filepath = "".join([path, 'k25_',date,'_', filename])
# # '..\\_data\\_save\\MCP\\k25_0117_1058_1000_0.3333.hdf5'


# mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1, save_best_only=True, filepath= filepath)        
# model.compile(loss='mae', optimizer='adam')                                                             
# es = EarlyStopping(monitor='val_loss', mode='min',patience=10, verbose= 20, restore_best_weights=True) 
# hist = model.fit(X_train, y_train, epochs=1000, batch_size=15, validation_split=0.1, callbacks=[es,mcp])
# # model.save("..\\_data\\_save\\keras25_MCP_3_save_model.hdf5")


model = load_model("..\\_data\\_save\\MCP\\k25_0117_1207_0083-2.4594-2.3641.hdf5")
print("=======================1.기본출력 ==============================")
loss = model.evaluate(X_test, y_test, verbose=0)
print("로스 : ", loss)

y_predict = model.predict(X_test)
# result = model.predict(X, verbose=0)

r2 = r2_score(y_test, y_predict)
print("R2스코어 :", r2)

# print("==============================================================")
# print(hist.history['val_loss'])
# print("==============================================================")


# restore_best_weights
# save_best_only


# True, True        Best
# True, False      모든에포별로 다 저장함
# False, False      전부 다 저장
# False, True       loss 좋은순으로 저장되긴 하지만 좋은 가중치는 아님  

