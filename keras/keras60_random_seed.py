from keras.models import Sequential
from keras.layers import Dense
import tensorflow as tf
import keras
import numpy as np
import random as rn
rn.seed(333)
tf.random.set_seed(132)       #텐서 2.9 에서는 먹히지만, 2.12에서는 안먹힘
np.random.seed(321)


print(keras.__version__)    # 2.9.0
print(tf.__version__)       # 2.9.0
print(np.__version__)       # 1.26.3

# 1. 데이터
X = np.array([1,2,3])
y = np.array([1,2,3])


# 2. 모델
model = Sequential()
model.add(Dense(5, input_dim=1)) 
model.add(Dense(5)) 
model.add(Dense(1)) 

# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(X, y, epochs=100)


# 4. 평가, 예측
loss = model.evaluate(X,y, verbose=[0])
print('loss : ', loss)
results = model.predict([4], verbose=0)
print('4의 예측값 : ', results)

