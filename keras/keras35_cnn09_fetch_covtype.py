from sklearn.datasets import fetch_covtype
import pandas as pd
import numpy as np
from keras.models import Sequential, Model
from keras.layers import Dense, Conv2D, Flatten, Dropout, GlobalAveragePooling2D, MaxPooling2D
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from keras.utils import to_categorical
from sklearn.preprocessing import OneHotEncoder
from keras. callbacks import EarlyStopping,ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, StandardScaler, RobustScaler

datasets = fetch_covtype()

X = datasets.data
y = datasets.target
# y = y.reshape(-1, 1)
# print(X.shape, y.shape)         # (581012, 54) (581012)
# print(pd.value_counts(y))       # 7
X = X.reshape(581012, 3, 3, 6)

# 2    283301
# 1    211840
# 3     35754
# 7     20510
# 6     17367
# 5      9493
# 4      2747

# 1. scikit-learn 방식
# y = OneHotEncoder(sparse=False).fit_transform(y)
# print(y.shape)      #(581012, 7)
# print(y)            

# 2. pandas 방식
# y = pd.get_dummies(y)
# print(y.shape)       #(581012, 7)


# 3. keras 방식
y = to_categorical(y)
y= y[:,1:]                 # 행렬 슬라이싱
# print(y.shape)          # (581012, 7)
# print(y)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35, shuffle=True, random_state=3, stratify=y)

################    MinMaxScaler    ##############################
# mms = MinMaxScaler()
# mms.fit(X_train)
# X_train = mms.transform(X_train)
# X_test = mms.transform(X_test)

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

# print(X_train.shape, X_test.shape)      # (464809, 54) (116203, 54)
# print(y_train.shape, y_test.shape)      # (464809, 7) (116203, 7)


# model = Sequential()
# model.add(Dense(19, input_dim =54, activation='relu'))
# model.add(Dense(97))
# model.add(Dense(9))
# model.add(Dense(21, activation='sigmoid'))
# model.add(Dense(9))
# model.add(Dense(97))
# model.add(Dropout(0.4))
# model.add(Dense(19))
# model.add(Dense(7, activation='softmax'))

# i1 = Input(shape = (54,))
# d1 = Dense(19,activation='relu')(i1)      
# d2 = Dense(97)(d1)
# d3 = Dense(9)(d2)
# d4 = Dense(21,activation='sigmoid')(d3)
# d5 = Dense(9)(d4)
# d6 = Dense(97)(d5)
# drop1 = Dropout(0.2)(d6)
# o1 = Dense(7,activation='softmax')(drop1)
# model = Model(inputs = i1, outputs = o1)

model = Sequential()                    
model.add(Conv2D(19, kernel_size=(3, 3), input_shape=(3, 3, 6), activation='swish', strides=2, padding='same'))   
model.add(Conv2D(97, (4, 4), activation='swish',strides=2, padding='same'))                         
model.add(Conv2D(500, (3, 3), activation='swish', strides=2, padding='same'))              
model.add(GlobalAveragePooling2D())  
model.add(Dense(124, activation='swish'))
model.add(Dropout(0.3)) 
model.add(Dense(48, activation='swish'))
model.add(Dense(7, activation='softmax'))


import datetime
date = datetime.datetime.now()
# print(date)                     # 2024-01-17 10:52:59.857389
date = date.strftime("%m%d_%H%M")                   # _는 str      
# print(date)                     # 0117_1058
# print(type(date))               # <class 'str'>

path = "..\\_data\\_save\\MCP\\"
filename = '{epoch:05d}-{acc:.4f}-{loss:.4f}.hdf5'            # 04d : 4자리 정수표현, 4f : 소수4번째자리까지 표현, 예) 1000_0.3333.hdf5
filepath = "".join([path, 'k28_09_fetch_covtype',date,'_', filename])

# mcp = ModelCheckpoint(monitor='val_loss', mode='min', verbose=1, save_best_only=True, filepath=filepath)    
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics='acc')
es = EarlyStopping(monitor='val_loss', mode='min', patience=10, restore_best_weights=True, verbose=1)
hist = model.fit(X_train, y_train, epochs=100, batch_size=1000, validation_split=0.1, callbacks=[es], verbose=1)

results = model.evaluate(X_test, y_test)
print("로스 : ", results[0])
print("ACC : ", results[1])

y_predict = model.predict(X_test)
y_test = np.argmax(y_test, axis=1)
y_predict = np.argmax(y_predict, axis=1)

acc = accuracy_score(y_test, y_predict)
print("accuracy_score : ", acc)


# 로스 :  0.6938604116439819
# ACC :  0.701832115650177
# accuracy_score :  0.7018321385850623

# 로스 :  0.5239526033401489
# ACC :  0.7768474221229553
# accuracy_score :  0.7768474135779627








# MinMaxScaler
# 로스 :  0.4319405257701874
# ACC :  0.819133996963501
# accuracy_score :  0.8191340267020727

# MaxAbsScaler
# 로스 :  0.43206891417503357
# ACC :  0.8199257254600525
# accuracy_score :  0.8199257456172703
# StandardScaler
# 로스 :  0.4352737367153168
# ACC :  0.8203535676002502
# accuracy_score :  0.8203535688820044

# RobustScaler
# 로스 :  0.4073517620563507
# ACC :  0.8323080539703369
# accuracy_score :  0.8323080327506085



# 로스 :  0.17327208817005157
# ACC :  0.9312630891799927
# accuracy_score :  0.9312630621327236