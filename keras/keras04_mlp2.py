# [실습]

import numpy as np
from keras.models import Sequential
from keras.layers import Dense

# 1. 데이터

x = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],                              
             [1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.5, 1.4, 1.3],
             [9,8,7,6,5,4,3,2,1,0]])                 

y = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
print(x)
print(x.shape, y.shape)     # (3, 10) (10,)
x = x.T
print(x.shape)              # (10, 3)


# 2. 모델구성
model = Sequential()
model.add(Dense(16,input_dim=3))
model.add(Dense(9))
model.add(Dense(23))
model.add(Dense(7))
model.add(Dense(21))
model.add(Dense(1))


# 3. 컴파일, 훈련
model.compile(loss='mse',optimizer='adam')
model.fit(x, y, epochs=100, batch_size=1)

# 4. 평가, 예측
loss = model.evaluate(x,y)
result = model.predict([[10,1.3,0]])
print("로스 : ", loss)
print("[10,1.3,0]의 예측값 : ", result)

# epochs=100, batch_size=2
#  로스 :  2.3695658455835655e-05
# [10,1.3,0]의 예측값 :  [[10.00682]]

# epochs=100, batch_size=1
# 로스 :  6.956016932235798e-06
# [10,1.3,0]의 예측값 :  [[9.999982]]

# epochs=10000, batch_size=3
# 로스 :  5.36601882719645e-12
# [10,1.3,0]의 예측값 :  [[10.000001]]

# epochs=10000, batch_size=2
# 로스 :  1.8332002040509499e-13
# [10,1.3,0]의 예측값 :  [[10.]]


