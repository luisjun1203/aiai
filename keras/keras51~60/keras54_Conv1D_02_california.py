from sklearn.datasets import fetch_california_housing
from keras.models import Sequential, Model
from keras.layers import Dense, Conv2D, Flatten, Dropout, GlobalAveragePooling2D, MaxPooling2D,LSTM, Conv1D
import numpy as np
from sklearn.model_selection import train_test_split
import time
import numpy as np
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, StandardScaler, RobustScaler
from keras.callbacks import EarlyStopping, ModelCheckpoint

datasets = fetch_california_housing()
X = datasets.data
y = datasets.target

# print(X.shape)  # (20640, 8)

X = X.reshape(20640, 4, 2)
print(X.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, shuffle=True, random_state=33)

# mas = MaxAbsScaler()
# mas.fit(X_train)
# X_train = mas.transform(X_train)
# X_test = mas.transform(X_test)



# model = Sequential()
# model.add(Dense(21,input_dim=8))
# model.add(Dense(7))
# model.add(Dense(18))
# # model.add(Dropout(0.5))
# model.add(Dense(12))
# model.add(Dense(1))

# i1 = Input(shape = (8,))
# d1 = Dense(21)(i1)      
# d2 = Dense(7)(d1)
# d3 = Dense(18)(d2)
# drop1 = Dropout(0.5)(d3)
# d4 = Dense(12)(d3)
# o1 = Dense(1)(drop1)
# model = Model(inputs = i1, outputs = o1)
model= Sequential()
model.add(Conv1D(filters=19,kernel_size=3, input_shape= (4,2)))                                                      
model.add(Flatten())
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
filepath = "".join([path, 'k28_02_california_',date,'_', filename])
                        
# mcp = ModelCheckpoint(monitor='val_loss', mode='min', verbose=1, save_best_only=True, filepath=filepath)    
model.compile(loss='mse', optimizer='adam')                                                             # early stopping 개념, min,max, auto
es = EarlyStopping(monitor='val_loss', mode='min',patience=100, verbose= 1, restore_best_weights=True) 
hist = model.fit(X_train, y_train, epochs=200, batch_size=400, validation_split=0.2, callbacks=[es])
# end_time = time.time()

loss = model.evaluate(X_test, y_test)
print("로스 : ", loss)

y_predict = model.predict(X_test)
result = model.predict(X)

from sklearn.metrics import r2_score, mean_squared_error
r2 = r2_score(y_test, y_predict)
def RMSE(aaa, bbb):
    np.sqrt(mean_squared_error(aaa, bbb))
    return np.sqrt(mean_squared_error(aaa, bbb))
rmse = RMSE(y_test, y_predict)

print("R2스코어 : ", r2)
# print("걸린시간 : ", round(end_time - start_time, 3), "초")
print("RMSE : ", rmse)

# 로스 :  0.5245388150215149
# R2스코어 :  0.6191872009775989
# RMSE :  0.7242504869844772

#RNN
# 로스 :  0.40590542554855347
# R2스코어 :  0.7053144620181305
# RMSE :  0.6371070914286465

#Conv1D

# 로스 :  0.5646118521690369
# R2스코어 :  0.5900942522523298
# RMSE :  0.7514066455288203
