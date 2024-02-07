# keras09_1_boston.py



import warnings
warnings.filterwarnings('ignore')

from sklearn.datasets import load_boston
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten ,GlobalAveragePooling2D, LSTM, Conv1D
import numpy as np
from sklearn.model_selection import train_test_split
import time
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, StandardScaler, RobustScaler
from keras.callbacks import EarlyStopping, ModelCheckpoint

datasets = load_boston()
# print(datasets)

X = datasets.data
y = datasets.target
# print(X.shape, y.shape)  # (506,13) (506,)

X = X.reshape(506, 13, 1)


# print(X.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, shuffle=True, random_state=20)

################    MinMaxScaler    ##############################
# mms = MinMaxScaler()
# mms.fit(X_train)
# X_train = mms.transform(X_train)
# X_test = mms.transform(X_test)

# X_train = X_train/506         
# X_test = X_test/506

################    StandardScaler    ##############################

# sts = StandardScaler()
# sts.fit(X_train)
# X_train = sts.transform(X_train)
# X_test = sts.transform(X_test)

# print(X_train)
# print(X_test)

# ################    MaxAbsScaler    ##############################
# mas = MaxAbsScaler()
# mas.fit(X_train)
# X_train = mas.transform(X_train)
# X_test = mas.transform(X_test)


# ################    RobustScaler    ##############################
# rbs = RobustScaler()
# rbs.fit(X_train)
# X_train = rbs.transform(X_train)
# X_test = rbs.transform(X_test)



# print(np.min(X_train))      # 0.0
# print(np.min(X_test))       # 0.0
# print(np.max(X_train))      # 1.0
# print(np.max(X_test))       # 1.210017220702162

# model = Sequential()
# model.add(Dense(19,input_dim=13,activation='relu'))
# model.add(Dense(97,activation='relu'))
# model.add(Dense(9,activation='relu'))
# model.add(Dense(21,activation='relu'))
# model.add(Dense(28,activation='relu'))
# model.add(Dense(1))


model = Sequential()
# model.add(LSTM(19, return_sequences=True,          
#                input_length = 13, input_dim = 1, activation='relu'))         
# model.add(LSTM(97, )) 
model.add(Conv1D(filters=19,kernel_size=3, input_shape= (13,1)))                                                      
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
filepath = "".join([path, 'k26_boston_',date,'_', filename])

# mcp = ModelCheckpoint(monitor='val_loss', mode='min', verbose=1, save_best_only=True, filepath=filepath)    
model.compile(loss='mse', optimizer='adam')                                                             # early stopping 개념, min,max, auto
es = EarlyStopping(monitor='val_loss', mode='min',patience=100, verbose= 1, restore_best_weights=True) 
# start_time = time.time()
model.fit(X_train, y_train, epochs=1500, batch_size=15, validation_split=0.1, callbacks=[es] )
# end_time = time.time()



loss = model.evaluate(X_test, y_test)
print("로스 : ", loss)
y_predict = model.predict(X_test)

# result = model.predict(X)

print(y_test.shape) # (76,)
print(y_predict.shape)  # (76, 1)

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print("R2스코어 : ", r2)

# R2스코어 :  0.7171061815697353

#rnn
# R2스코어 :  0.7246546233581659

#Conv1D
# R2스코어 :  0.7933664937783182
