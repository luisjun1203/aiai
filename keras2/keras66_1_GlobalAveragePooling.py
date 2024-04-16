# 각 필터마다 GlobalAveragePooling  적용
# 필터별로 특성저장
# 연산량 감소, 통상적으로 flatten보다 성능 좋음(하지만 둘 다 해봐)

import numpy as np
from keras.datasets import mnist
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, GlobalAveragePooling2D
from sklearn.preprocessing import OneHotEncoder
import time
from keras.callbacks import EarlyStopping

# 1. 데이터

(X_train, y_train), (X_test, y_test) = mnist.load_data()
# print(X_train.shape, y_train.shape)     # (60000, 28, 28) (60000,)
# print(X_test.shape, y_test.shape)       # (10000, 28, 28) (10000,)              # cnn사용할때는 reshape 해줘야함
# print(X_train)
print(X_train[9])
print(y_train[9])
# print(np.unique(y_train, return_counts=True)) #(array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=uint8), array([5923, 6742, 5958, 6131, 5842, 5421, 5918, 6265, 5851, 5949],dtype=int64))     
print(pd.value_counts(y_test))

# import matplotlib.pyplot as plt
# plt.imshow(X_train[9], 'gray')
# plt.show()


X_train = X_train.reshape(60000, 28, 28, 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)
print(X_train.shape)            # (60000, 28, 28, 1)
print(X_test.shape)             # (10000, 28, 28, 1)
# print(y_test)
# print(y_train)

y_test = y_test.reshape(-1,1)
y_train = y_train.reshape(-1,1)

ohe = OneHotEncoder(sparse=False)
y_train = ohe.fit_transform(y_train)
y_test = ohe.fit_transform(y_test)


print(y_test)
print(y_train)

# print(y_train, y_test)
# 2. 모델 구성
                                        # (kenal높이 x kernel너비 x 입력채널수+1) x filters     +1은 bias
model = Sequential()                    # input channel x output channel x kernel_size + output channel : 1x19x(2x2)+19
model.add(Conv2D(19, kernel_size=(2, 2),input_shape = (28, 28, 1), strides=2, padding='same'))   #(N,27,27,19)             # 19 : filter( out layer 이름)
model.add(MaxPooling2D())
model.add(Conv2D(97, (3, 3), strides=2, padding='same')) #(N,25,25,97)          
model.add(Conv2D(9, (4, 4), strides=2, padding='same'))               #(N,22,22,9)
# model.add(Flatten())                       # (N, 22x22x9)                             # flatten : reshpae 할 필요없이 2d를 1d로 바꿔준다
model.add(GlobalAveragePooling2D())
model.add(Dense(21, activation='relu'))
model.add(Dense(10, activation= 'softmax'))

model.summary()


# # 3. 컴파일 , 훈련
# strat_time = time.time()
# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
# es = EarlyStopping(monitor='val_loss', mode='min', patience=100, verbose=20, restore_best_weights=True)
# model.fit(X_train, y_train, epochs=1000, batch_size=5000,verbose=2, validation_split=0.15, callbacks=[es])
# end_time = time.time()
# # print(X_train, X_test)

# # 4. 평가, 예측


# results = model.evaluate(X_test, y_test)
# print('loss' , results[0])
# print('acc', results[1])
# print("걸리시간 : ", round(end_time - strat_time, 3), "초")


# loss 0.7981159925460815
# acc 0.9863999772071838

# stride_padding 적용

# loss 0.22794955968856812
# acc 0.9791000270843506
# 걸리시간 :  72.562 초

#Maxpoling 적용

# loss 0.08957613259553909
# acc 0.9732999801635742
# 걸리시간 :  36.849 초

