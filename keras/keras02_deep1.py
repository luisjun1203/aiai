from keras.models import Sequential
from keras.layers import Dense
import numpy as np

#1. 데이터
x = np.array([1,2,3])
y = np.array([1,2,3])

#2. 모델 구성
model = Sequential()
model.add(Dense(3, input_dim=1))
model.add(Dense(20))
model.add(Dense(33))
model.add(Dense(47))
model.add(Dense(95))
model.add(Dense(15))
model.add(Dense(22))
model.add(Dense(723))
model.add(Dense(52))
model.add(Dense(54))
model.add(Dense(66))
model.add(Dense(58))
model.add(Dense(60))
model.add(Dense(85))
model.add(Dense(26))
model.add(Dense(1))




#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x,y, epochs=100)

#4. 평가, 훈련
loss = model.evaluate(x,y)
print("로스: ", loss)
result = model.predict([4])
print("4의 예측값: ", result)

# #로스:  0.0002875525096897036
# 1/1 [==============================] - 0s 104ms/step
# 4의 예측값:  [[ 4.0000896]]

