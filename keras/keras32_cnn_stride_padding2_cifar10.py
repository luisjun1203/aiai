from keras.datasets import cifar10
import numpy as np
from keras.datasets import mnist
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, Dropout, GlobalAveragePooling2D
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder,StandardScaler, RobustScaler
import time
from keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score

# acc = 0.77이상

(X_train, y_train), (X_test, y_test) = cifar10.load_data()

# print(X_test.shape, y_test.shape)       #(10000, 32, 32, 3) (10000, 1)
# print(X_train.shape, y_train.shape)     #(50000, 32, 32, 3) (50000, 1)

# print(np.unique(y_train,return_counts=True))
# (array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=uint8), array([5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000], dtype=int64))
# print(pd.value_counts(y_test))
# X_train = X_train.astype('float32')
# X_test = X_train.astype('float32')


# print(X_test)
# print(X_train)


# print(y_test.shape)
# print(y_train.shape)

# y_train = y_train.reshape(-1, 1)
# y_test = y_test.reshape(-1, 1)

# print(y_test.shape)
# print(y_train.shape)

ohe = OneHotEncoder(sparse=False)
y_train = ohe.fit_transform(y_train)
y_test = ohe.fit_transform(y_test)

# print(y_test)
# print(y_train)

##### minmaxScaler ######
# X_train = X_train/255           # 0~255까지 있는 데이터라 255로 나눠줌
# X_test = X_test/255


##### standardScaler #######
mean = np.mean(X_train, axis=(0, 1))     # 평균
std = np.std(X_train, axis=(0, 1))       # 표준편차
X_train = (X_train - mean) / std            
X_test = (X_test - mean) / std

##### standardScaler #######
# scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train.reshape(-1, 32*32*3)).reshape(-1, 32, 32, 3)
# X_test = scaler.transform(X_test.reshape(-1, 32*32*3)).reshape(-1, 32, 32, 3)

# rbs = RobustScaler()
# X_train = rbs.fit_transform(X_train.reshape(-1, 32*32*3)).reshape(-1, 32, 32, 3)
# X_test = rbs.fit_transform(X_train.reshape(-1, 32*32*3)).reshape(-1, 32, 32, 3)

# print(X_train)
# print(X_test)

# model = Sequential()                    
# model.add(Conv2D(19, kernel_size=(2, 2),input_shape = (32, 32, 3),activation='relu'))   
# model.add(Conv2D(45, (3, 3 ),activation='relu'))                         
# model.add(Conv2D(14, (4, 4 ),activation='relu'))              
# model.add(Conv2D(9, (2, 2 ),activation='relu'))
# model.add(Conv2D(25, (3, 3 ),activation='relu'))                         
# model.add(Conv2D(12, (4, 4 ),activation='relu'))              
# model.add(Conv2D(9, (2, 2 ),activation='relu'))              
# model.add(Flatten())                                                  
# model.add(Dense(22,activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(30,activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(18,activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(10, activation= 'softmax'))

model = Sequential()                    
model.add(Conv2D(19, kernel_size=(3, 3), input_shape=(32, 32, 3),activation='swish'))   
model.add(Conv2D(97, (4, 4),activation='swish'))                         
model.add(Conv2D(210, (3, 3), strides=2, padding='same',activation='swish'))              
model.add(GlobalAveragePooling2D())  # Global Average Pooling 사용
model.add(Dense(81, activation='swish'))
model.add(Dropout(0.3))  # 낮은 Dropout 비율 사용
model.add(Dense(48, activation='swish'))
model.add(Dense(10, activation='softmax'))
model.summary()


strat_time = time.time()
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
es = EarlyStopping(monitor='val_loss', mode='min', patience=50, verbose=2, restore_best_weights=True)
model.fit(X_train, y_train, epochs=1997, batch_size=97,verbose=2, validation_split=0.15, callbacks=[es])
end_time = time.time()
# print(X_train, X_test)

# 4. 평가, 예측

results = model.evaluate(X_test, y_test)
y_predict = model.predict(X_test)
y_test = np.argmax(y_test, axis=1)
y_predict = np.argmax(y_predict, axis=1)
acc = accuracy_score(y_test, y_predict)
# print(y_test)

print('loss' , results[0])
print('acc', results[1])
print("걸리시간 : ", round(end_time - strat_time, 3), "초")
print("accuracy_score : ", acc)


# loss 0.7100104093551636
# acc 0.7717999815940857
# 걸리시간 :  1785.19 초
# accuracy_score :  0.7718
















