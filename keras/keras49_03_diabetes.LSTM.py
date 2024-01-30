from sklearn.datasets import load_diabetes
from keras.models import Sequential, Model
from keras.layers import Dense, Conv2D, Flatten, Dropout, GlobalAveragePooling2D, MaxPooling2D,LSTM
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import time
import numpy as np
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, StandardScaler, RobustScaler

datasets = load_diabetes()
X = datasets.data
y = datasets.target          


print(X.shape)  #(442,10)
X = X.reshape(442,10,1)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, shuffle=True, random_state=1226)           # 1226 713

# mms = MinMaxScaler()
# mms.fit(X_train)
# X_train = mms.transform(X_train)
# X_test = mms.transform(X_test)


# model = Sequential()
# model.add(Dense(8,input_dim=10))
# model.add(Dense(16))
# model.add(Dense(24))
# model.add(Dropout(0.5))
# model.add(Dense(16))
# model.add(Dense(8))
# model.add(Dense(1))

# i1 = Input(shape = (10,))
# d1 = Dense(8)(i1)      
# d2 = Dense(16)(d1)
# d3 = Dense(24)(d2)
# drop1 = Dropout(0.5)(d3)
# d4 = Dense(16)(drop1)
# d5 = Dense(8)(d4)
# o1 = Dense(1)(d5)
# model = Model(inputs = i1, outputs = o1)

model = Sequential()
model.add(LSTM(19, return_sequences=True,          
               input_length = 10, input_dim = 1, activation='relu'))         
model.add(LSTM(97, ))                                                       
model.add(Dense(9, activation='swish'))
model.add(Dense(21, activation='swish'))
model.add(Dense(16, activation='swish'))
model.add(Dense(21, activation='swish'))
model.add(Dense(1))
model.summary()

import datetime
date = datetime.datetime.now()
print(date)                     # 2024-01-17 10:52:59.857389
date = date.strftime("%m%d_%H%M")                   # _는 str      
print(date)                     # 0117_1058
print(type(date))               # <class 'str'>

path = "..\\_data\\_save\\MCP\\"
filename = '{epoch:05d}-{val_loss:.4f}-{loss:.4f}.hdf5'            # 04d : 4자리 정수표현, 4f : 소수4번째자리까지 표현, 예) 1000_0.3333.hdf5
filepath = "".join([path, 'k28_03_diabetes_',date,'_', filename])


# mcp = ModelCheckpoint(monitor='val_loss', mode='min', verbose=1, save_best_only=True, filepath=filepath)    
model.compile(loss='mse', optimizer='adam')
es = EarlyStopping(monitor='val_loss', mode='min',patience=100, verbose= 1, restore_best_weights=True) 
start_time = time.time()
hist = model.fit(X_train, y_train, epochs=500, batch_size=10, validation_split=0.2, callbacks=[es])
end_time = time.time()

loss = model.evaluate(X_test, y_test)
y_predict = model.predict(X_test)
r2 = r2_score(y_test, y_predict)
def RMSE(aaa, bbb):
    np.sqrt(mean_squared_error(aaa, bbb))
    return np.sqrt(mean_squared_error(aaa, bbb))
rmse = RMSE(y_test, y_predict)
print("로스 : ", loss)

print("걸린시간 : ",round(end_time - start_time, 3), "초")

# 걸린시간 :  7.659 초
# 걸린시간 :  5.85 초

# 로스 :  2133.96044921875
# 걸린시간 :  46.361 초

# rnn
# 로스 :  3003.519775390625
# 걸린시간 :  42.962 초