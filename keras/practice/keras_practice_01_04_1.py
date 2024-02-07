from keras.models import Sequential
from keras.layers import Dense
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import time


# 1. 데이터
X = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20])
y = np.array([1,2,4,3,5,7,9,3,8,12,13,8, 14,15, 9, 6,17,23,21,20])

# print(X.shape)
# print(y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, shuffle=True, random_state=42)

model = Sequential()
model.add(Dense(3, input_dim=1))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
start_time = time.time()
model.fit(X_train, y_train, epochs=100, batch_size=2)
end_time = time.time()

loss = model.evaluate(X_test, y_test)
print("로스 : ", loss)
y_predict = model.predict(X_test)
result = model.predict([X])

r2 = r2_score(y_test, y_predict)
print("R2스코어 : ", r2)

import  matplotlib.pyplot as plt

plt.scatter(X, y)
plt.plot(X, result, color='pink')
plt.show()

