import numpy as np
from keras.models import Sequential
from keras.layers import Dense

# pip install numpy     # cmd->pip list : 확인용

# 1. 데이터
# x1 = np.array([range(10)])    # range : n-1
# print(x1)                     # [[0 1 2 3 4 5 6 7 8 9]]
# print(x1.shape)               # (1, 10)

# x2 = np.array([range(1, 10)]) # range : n-1
# print(x2)                     # [[1 2 3 4 5 6 7 8 9]]
# print(x2.shape)               # (1, 9)

x = np.array([range(10), range(21, 31), range(201, 211)])
# print(x)
# print(x.shape)                  

x = x.T
# print(x)                        
# #print(x.shape)                  # (10, 3)


y = np.array([range(1,11), [1,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9],
              range(9,-1,-1)])        # []: 두개이상은 list 중요

print(y)
print(y.shape)         
y = y.T

# 예측 : [10, 31, 211]

model = Sequential()
model.add(Dense(4, input_dim=3))
model.add(Dense(25))
model.add(Dense(15))
model.add(Dense(23))
model.add(Dense(12))
model.add(Dense(3))

model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=100, batch_size=1)

loss = model.evaluate(x,y)
result = model.predict([[10,31,211]])
print("로스 : ", loss)
print("[10,31,211]의 예측값 : ", result)

# epochs=100, batch_size=1
# 로스 :  6.277859308789768e-12
# [10,31,211]의 예측값 :  [[11.000004   1.9999965 -1.0000002]]
