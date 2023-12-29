# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense
import numpy as np
from keras.models import Sequential
from keras.layers import Dense


#1.데이터 
x = np.array([1,2,3,4,5,6])
y = np.array([1,2,3,5,4,6])

#2.모델구성
####[실습]100에포에 01_1번과 같은 결과를 얻어보자!
model = Sequential()
model.add(Dense(3,input_dim=1))
model.add(Dense(40))
model.add(Dense(10))
model.add(Dense(30))
model.add(Dense(20))
model.add(Dense(50))
model.add(Dense(80))
model.add(Dense(70))
model.add(Dense(90))
model.add(Dense(100))
model.add(Dense(1))


#3.컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=100)

#4.평가, 예측
loss = model.evaluate(x,y)
print("로스: ", loss)
result = model.predict([7])
print("7의 예측값: ", result)

# # 로스:  0.32390865683555603
# 1/1 [==============================] - 0s 98ms/step
# 7의 예측값:  [[6.816275]]