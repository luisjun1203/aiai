# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense
import numpy as np
from keras.models import Sequential
from keras.layers import Dense


#1.데이터 
x = np.array([1,2,3,4,5,6])
y = np.array([1,2,3,5,4,6])

#2.모델구성
model = Sequential()
model.add(Dense(1,input_dim=1))


#3.컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=10000)

#4.평가, 예측
loss = model.evaluate(x,y)
print("로스: ", loss)
result = model.predict([1,2,3,4,5,6,7])
print("7의 예측값: ", result)

# 로스:  0.32381191849708557
# 1/1 [==============================] - 0s 38ms/step
# 7의 예측값:  [[1.1455028]
#  [2.0875206]
#  [3.0295382]
#  [3.971556 ]
#  [4.9135737]
#  [5.8555913]
#  [6.7976093]]
#  에포: 10000
