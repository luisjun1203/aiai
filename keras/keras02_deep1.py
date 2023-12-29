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
model.add(Dense(234))
model.add(Dense(474))
model.add(Dense(629))
model.add(Dense(100))




#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x,y, epochs=100)

#4. 평가, 훈련
loss = model.evaluate(x,y)
print("로스: ", loss)
result = model.predict([4])
print("4의 예측값: ", result)

# #로스:  1.5958017684170045e-05
# 1/1 [==============================] - 0s 104ms/step
# 4의 예측값:  [[ 3.9999795]]

