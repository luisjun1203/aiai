import numpy as np
from keras.models import Sequential
from keras.layers import Dense

x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([1,2,3,4,6,5,7,8,9,10])

x = x.T
y = y.T


model = Sequential()
model.add(Dense(4,input_dim=1))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=500, batch_size=1)

loss = model.evaluate(x,y)
result = model.predict([11000, 7])
print("로스 : ", loss)
print("[11000000]의 예측값 : ", result)



# 로스 :  3.197442310920451e-13
# [11000000]의 예측값 :  [[11000001.]]