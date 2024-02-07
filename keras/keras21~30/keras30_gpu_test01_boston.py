#DropOut 


# keras09_1_boston.py
import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import r2_score
from sklearn.datasets import load_boston
from keras.models import Sequential, load_model,Model
from keras.layers import Dense, Dropout , Input                       # 직전 레이어의 노드를 설정한 비율만큼 무작위로 빼준다 -> 하지만 evaluate단계에서는 의미가 없다 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, StandardScaler, RobustScaler
from keras.callbacks import EarlyStopping, ModelCheckpoint 
import time

datasets = load_boston()

X = datasets.data
y = datasets.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, shuffle=True, random_state=20)

mms = MinMaxScaler()
mms.fit(X_train)
X_train = mms.transform(X_train)
X_test = mms.transform(X_test)

# # 2. 모델 구성
# model = Sequential()
# model.add(Dense(19,input_dim=13,activation='relu'))
# model.add(Dense(97,activation='relu'))
# model.add(Dense(9,activation='relu'))
# model.add(Dense(21,activation='relu'))
# model.add(Dense(28,activation='relu'))
# # model.add(Dropout(0.5))
# model.add(Dense(1))

i1 = Input(shape = (13,))
d1 = Dense(19,activation='relu')(i1)      
d2 = Dense(97,activation='relu')(d1)
d3 = Dense(9,activation='relu')(d2)
d4 = Dense(21,activation='relu')(d3)
d5 = Dense(16,activation='relu')(d4)
d6 = Dense(21,activation='relu')(d5)
drop1 = Dropout(0.5)(d6)
o1 = Dense(1)(drop1)
model = Model(inputs = i1, outputs = o1)


import datetime
date = datetime.datetime.now()
print(date)                     # 2024-01-17 10:52:59.857389
date = date.strftime("%m%d_%H%M")                   # _는 str      
print(date)                     # 0117_1058
print(type(date))               # <class 'str'>

path = "..\\_data\\_save\\MCP\\"
filename = '{epoch:04d}-{val_loss:.4f}-{loss:.4f}.hdf5'            # 04d : 4자리 정수표현, 4f : 소수4번째자리까지 표현, 예) 1000_0.3333.hdf5
filepath = "".join([path, 'k28_01_boston_',date,'_', filename])


# mcp = ModelCheckpoint(monitor='loss', mode='min', verbose=1, save_best_only=True, filepath= filepath)        
model.compile(loss='mae', optimizer='adam')                     
start = time.time()                                        
es = EarlyStopping(monitor='loss', mode='min',patience=100, verbose= 20, restore_best_weights=True) 
hist = model.fit(X_train, y_train, epochs=1000, batch_size=15, validation_split=0.1, callbacks=[es])
end = time.time()

print("=======================1.기본출력 ==============================")
loss = model.evaluate(X_test, y_test, verbose=0)
print("로스 : ", loss)

y_predict = model.predict(X_test)
# result = model.predict(X, verbose=0)

r2 = r2_score(y_test, y_predict)
print("R2스코어 :", r2)
print("걸린시간 : ",round(end - start, 3), "초")


# 걸린시간 :  52.388 초
# 걸린시간 :  33.221 초