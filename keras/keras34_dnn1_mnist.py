# keras_31_cnn3_연산량 계산.py


import numpy as np
from keras.datasets import mnist
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder
from keras.callbacks import EarlyStopping

# 1. 데이터

(X_train, y_train), (X_test, y_test) = mnist.load_data()
# print(X_train.shape, y_train.shape)     # (60000, 28, 28) (60000,)
# print(X_test.shape, y_test.shape)       # (10000, 28, 28) (10000,)              # cnn사용할때는 reshape 해줘야함
# print(X_train)
# print(X_train[9])
# print(y_train[9])
# # print(np.unique(y_train, return_counts=True)) #(array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=uint8), array([5923, 6742, 5958, 6131, 5842, 5421, 5918, 6265, 5851, 5949],dtype=int64))     
# print(pd.value_counts(y_test))

# import matplotlib.pyplot as plt
# plt.imshow(X_train[9], 'gray')
# plt.show()


X_train = X_train.reshape(60000, 28*28*1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1]*X_test.shape[2]*1)
# print(X_train.shape)     # (60000, 784)       # (60000, 28, 28, 1)
print(X_test.shape)        # (10000, 784)     # (10000, 28, 28, 1)



# # print(y_train, y_test)
# 2. 모델 구성
                                       
model = Sequential()                    
model.add(Dense(19, input_shape=(784, ) , activation='swish'))
model.add(Dropout(0.2))
model.add(Dense(97, activation='swish'))
model.add(Dropout(0.3))
model.add(Dense(9, activation='swish'))
model.add(Dense(21, activation='swish'))
model.add(Dropout(0.2))
model.add(Dense(5, activation='swish'))
model.add(Dropout(0.3))
model.add(Dense(31, activation='swish'))
model.add(Dense(10, activation='softmax'))


# model.summary()


# 3. 컴파일 , 훈련

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['acc'])
es = EarlyStopping(monitor='val_loss', mode='min', patience=97, verbose=2, restore_best_weights=True)
model.fit(X_train, y_train, epochs=1000, batch_size=300,verbose=2, validation_split=0.15, callbacks=[es])

# print(X_train, X_test)

# 4. 평가, 예측


results = model.evaluate(X_test, y_test)
print('loss' , results[0])
print('acc', results[1])


# stride_padding 적용
# loss 0.8981159925460815
# acc 0.9563999772071838


# dnn
# loss 0.7266284823417664
# acc 0.7307999730110168


