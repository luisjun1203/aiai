import numpy as np
import tensorflow as tf
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
tf.random.set_seed(777)
np.random.seed(777)
print(tf.__version__)   # 2.9.0

# 1 . 데이터
X = np.array([1, 2])
y = np.array([1, 2])


# 2 . 모델

model = Sequential()
model.add(Dense(2, input_dim=1))
# model.add(Dense(2))
model.add(Dense(1))



################################################################################
model.trainable = False # ★★★
#################################################################################
print("=================================================================")
print(model.weights)
print("=================================================================")

# 3. 컴파일, 훈련
model.compile(loss = 'mse', optimizer='adam')
model.fit(X, y, batch_size=1, epochs=1000, verbose=0)

# 4. 평가, 예측
y_predict = model.predict(X)
print(y_predict)





