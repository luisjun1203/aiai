import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

#1. 데이터

X = np.array(range(1, 17))
y = np.array(range(1, 17))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, shuffle=True, random_state=3 )
# print(X_test)       # [14 15 16]
# print(X_train)      # [ 1  2  3  4  5  6  7  8  9 10 11 12 13]
# print(y_train)      # [ 1  2  3  4  5  6  7  8  9 10 11 12 13]   
# print(y_test)       # [14 15 16]

#2. 모델구성
model = Sequential()
model.add(Dense(4, input_dim=1))
model.add(Dense(16))
model.add(Dense(64))
model.add(Dense(4))
model.add(Dense(1))

#3 .컴,훈
model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, epochs=200, batch_size=1, validation_split=0.3, verbose=1)

loss = model.evaluate(X_test, y_test)
result = model.predict([17])
print("로스 : ", loss)
print("[17]의 예측값 : ", result)





