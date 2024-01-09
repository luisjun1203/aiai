import numpy as np
from keras.models import Sequential
from keras.layers import Dense



#1 .데이터

X = np.array(range(1, 18))
y = np.array(range(1, 18))

X_val = X[10:13]
y_val = y[10:13]

X_train = X[0:-3]
y_train = y[:14]

X_test = X[-3:]
y_test = y[14:]

# print(X_train)        #[ 1  2  3  4  5  6  7  8  9 10 11 12 13 14]
# print(X_test)         #[15 16 17]
# print(y_train)        #[ 1  2  3  4  5  6  7  8  9 10 11 12 13 14]
# print(y_test)         #[15 16 17]
# print(X_val)          #[11 12 13]
# print(y_val)          #[11 12 13]

model = Sequential()
model.add(Dense(8, input_dim=1))
model.add(Dense(16))
model.add(Dense(8))
model.add(Dense(16))
model.add(Dense(8))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, epochs=100, batch_size=1, validation_data=(X_val, y_val))

loss = model.evaluate(X_test, y_test)
result = model.predict([18])
print("로스 : ", loss)
print("[18]의 예측값: ", result)



# 로스 :  2.1321586718414665e-09
# [18]의 예측값:  [[18.000053]]

# 로스 :  2.4253192770079535e-12
# [18]의 예측값:  [[18.000002]]





