import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split



#1 .데이터

X = np.array(range(1, 17))
y = np.array(range(1, 17))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, shuffle=False,)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, train_size=0.8 , shuffle = False,)

# print(X_train)      # [ 1  2  3  4  5  6  7  8  9 10]
# print(X_val)        # [11 12 13]
# print(X_test)       # [14 15 16]

model = Sequential()
model.add(Dense(4, input_dim=1))
model.add(Dense(16))
model.add(Dense(64))
model.add(Dense(4))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, epochs=200, batch_size=1, validation_data=(X_val, y_val), verbose=3)

loss = model.evaluate(X_test, y_test)
result = model.predict([17])
print("로스: ", loss)
print("[17]의 예측값 : ", result)

# 로스:  1.2126596385039767e-12
# [17]의 예측값 :  [[17.000002]]

# 로스:  1.515824466814808e-12
# [17]의 예측값 :  [[17.]]


















