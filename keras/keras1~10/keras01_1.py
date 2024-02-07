import tensorflow as tf # tensorflow를 땡겨오고, tf라고 줄여서 쓴다.
print(tf.__version__)   # 2.15.0
from tensorflow.keras.models import Sequential  #  tensorflow.keras.models에서 Sequential를 땡겨온다.
from tensorflow.keras.layers import Dense
import numpy as np


#1. 데이터
x = np.array([1,2,3])     # numpy 형식의 행렬
y = np.array([1,2,3])


#2. 모델구성
model = Sequential()                # 순차적인 모델을 만든다 model를 정의한다.
model.add(Dense(1, input_dim=1))    # 1은 y값 output, [input_dim=1]-x 한개의 데이터 덩어리

#3. 컴파일, 훈련
model.compile(loss='mse' , optimizer='adam')  # mse=두 값의 차이 (loss값은 항상 양수다) s=square'제곱하겠다' optimizer='adam' 이건 그냥 외우기
model.fit(x, y, epochs=12000)                    #최적의 웨이트 생성           # fit=훈련시킨다, epochs= 훈련 횟수 

#4. 평가, 예측
loss = model.evaluate(x, y)                     #loss를 평가한다
print("로스 : ", loss)
result = model.predict([4])
print("4의 예측값 : " , result)

# # 로스 :  0.0
# 1/1 [==============================] - 0s 69ms/step
# 4의 예측값 :  [[4.]]
