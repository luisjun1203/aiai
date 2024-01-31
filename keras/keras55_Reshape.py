import numpy as np
from keras.datasets import mnist
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, Reshape, Conv1D, LSTM
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder
import time
from keras.callbacks import EarlyStopping

# import tensorflow as tf
# print(tf.__version__)
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
                                        
model = Sequential()                    
model.add(Dense(9, input_shape=(28,28,1)))      # (N, 28, 28, 9)
model.add(Conv2D(10, (3, 3)))                   # (N,26,26,10)
model.add(Reshape(target_shape=(26*26,10)))     # (N, 676, 10)
model.add(Conv1D(15, 4))                        # (N, 673, 15)
model.add(LSTM(8, return_sequences=True))       # (N, 673, 8)
model.add(Conv1D(14, 2))                        # (N, 672, 14)
model.add(Dense(units=8))                       # (N, 672, 8)
model.add(Dense(7))                             # (N, 672, 7)
model.add(Flatten())                            # (N, 672*7)
model.add(Dense(6))                             # (N, 6)
model.add(Dense(10, activation= 'softmax'))     # (N, 10)

model.summary()


# 3. 컴파일 , 훈련
strat_time = time.time()
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
es = EarlyStopping(monitor='acc', mode='max', patience=100, verbose=20, restore_best_weights=True)
model.fit(X_train, y_train, epochs=1000, batch_size=5000,verbose=2, validation_split=0.15, callbacks=[es])
end_time = time.time()
# print(X_train, X_test)

# 4. 평가, 예측


results = model.evaluate(X_test, y_test)
print('loss' , results[0])
print('acc', results[1])
print("걸리시간 : ", round(end_time - strat_time, 3), "초")


# loss 0.7981159925460815
# acc 0.9863999772071838


# loss 0.1645084172487259
# acc 0.9711999893188477
# 걸리시간 :  334.828 초



