from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import tensorflow as tf
import keras
print("tf 버전 : ", tf.__version__)
print("keras 버전 : ", keras.__version__)

# 1. 데이터
x = np.array([1,2,3,4,5,6])
y = np.array([1,2,3,5,4,6])

# 2.모델구성
model = Sequential()
model.add(Dense(3,input_dim=1))
model.add(Dense(7))
model.add(Dense(12))
model.add(Dense(19))
model.add(Dense(16))
model.add(Dense(15))
model.add(Dense(18))
model.add(Dense(13))
model.add(Dense(8))
model.add(Dense(4))
model.add(Dense(1))


# 3.컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=100, batch_size=3)       # batch_size: 일괄 처리할 수 있는 데이터 사이즈 조절


# 4. 평가, 예측
loss = model.evaluate(x, y)
results = model.predict([7])
print("로스 : ", loss)
print("7의 예측값 : ", results)

# epochs=100, batch_size=6
# 로스 :  0.32386964559555054
# 7의 예측값 :  [[6.8162956]]