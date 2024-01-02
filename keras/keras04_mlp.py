import numpy as np
from keras.models import Sequential
from keras.layers import Dense

# 1. 데이터
x = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],                              # 행과열을 바꾸는 np (x = x.T)
             [1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.5, 1.4, 1.3]])                 

y = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

print(x.shape)  # (2, 10)
print(y.shape)  # (10, )
# x = x.T
x = x.transpose()

# [[1,1]], [2,1.1], [3,1.2], ... [10,1.3]]
print(x.shape)  # (2, 10)

# 2. 모델 구성
model = Sequential()
model.add(Dense(3, input_dim=2))
# 열, 컬럼, 속성, 특성, 차원 = 2 // 다 같은 의미다.
# (행무시, 열우선) <= 외우기
model.add(Dense(5))
model.add(Dense(9))
model.add(Dense(28))
model.add(Dense(32))
model.add(Dense(11))
model.add(Dense(1))


# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=100, batch_size = 2)


# 4. 평가, 예측
loss = model.evaluate(x,y)
print("로스 : ", loss)
result = model.predict([[10, 1.3]])                # shape 중요 
print("[10, 1.3]의 예측값 : ", result)

# [실습] : 소수 2째 자리까지 맞추기

# epochs = 200, batch_size = 2
# 로스 :  9.639318159315735e-05
# 1/1 [==============================] - 0s 77ms/step
# [10, 1.3]의 예측값 :  [[10.005851]]

# epochs = 100, batch_size = 2
# 로스 :  0.003710194258019328
# 1/1 [==============================] - 0s 85ms/step
# [10, 1.3]의 예측값 :  [[10.007009]]
