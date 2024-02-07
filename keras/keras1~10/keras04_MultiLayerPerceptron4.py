import numpy as np
from keras.models import Sequential
from keras.layers import Dense



x = np.array([range(10)])
print(x)
print(x.shape)
x = x.T
print(x)
print(x.shape)

y = np.array([range(1,11),np.arange(1, 2, 0.1),range(9,-1,-1)])       
                            # []: 두개이상은 list 중요

print(y)
print(y.shape)         
y = y.T

model = Sequential()
model.add(Dense(4,input_dim=1))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(3))

model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=100, batch_size=1)

loss = model.evaluate(x,y)
result = model.predict([10])
print("로스 : ", loss)
print("[10]의 예측값 : ", result)

# epochs=100, batch_size=1
# 로스 :  6.400805992005931e-13
# [10]의 예측값 :  [[11.         2.        -0.9999992]]
