# keras_31_cnn3_연산량 계산.py


import numpy as np
from keras.datasets import mnist
import pandas as pd
from keras.models import Sequential, Model
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout,Input
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
                                       
# model = Sequential()                    # input channel x output channel x kernel_size + output channel : 1x19x(2x2)+19
# model.add(Conv2D(19, kernel_size=(2, 2),input_shape = (28, 28, 1), strides=2, padding='same'))   #(N,27,27,19)             # 19 : filter( out layer 이름)
# model.add(MaxPooling2D())
# model.add(Conv2D(97, (3, 3), strides=2, padding='same')) #(N,25,25,97)          
# model.add(Conv2D(9, (4, 4), strides=2, padding='same'))               #(N,22,22,9)
# model.add(Flatten())                       # (N, 22x22x9)                             # flatten : reshpae 할 필요없이 2d를 1d로 바꿔준다
# model.add(Dense(21, activation='relu'))
# model.add(Dense(10, activation= 'softmax'))

i1 = Input(shape = (28, 28, 1))
conv1 = Conv2D(19, kernel_size=(2,2), activation='swish', strides=2, padding='same')(i1)      
max1 = MaxPooling2D(97,activation='swish')(conv1)
conv2 = Conv2D(9, (3.3), activation='relu')(max1)
conv3 = Conv2D(21, (4,4), activation='relu')(conv2)
flat1 = Flatten()(conv3)
d1 = Dense(21,activation='relu')(flat1)
d2 = Dense(21,activation='relu')(d1)
o1 = Dense(10,activation='softmax')(d2)
model = Model(inputs = i1, outputs = o1)


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