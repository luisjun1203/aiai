import numpy as np
from keras.models import Sequential
from keras.layers import Dense

# x = np.array([1,2,3,4,5,6,7,8,9,10])
# y = np.array([1,2,3,4,6,5,7,8,9,10])

x_train = np.array([1,2,3,4,5,6,7])         # 훈련
y_train = np.array([1,2,3,4,6,5,7])

x_test = np.array([8,9,10])                 # 테스트
y_test = np.array([8,9,10])


model = Sequential()
model.add(Dense(4,input_dim=1))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=500, batch_size=1)

loss = model.evaluate(x_test, y_test)
result = model.predict([11000, 7])
print("로스 : ", loss)
print("[11000000]의 예측값 : ", result)


# 1/1 [==============================] - 0s 81ms/step - loss: 0.0036
# 1/1 [==============================] - 0s 65ms/step
# 로스 :  0.003616855712607503
# [11000000]의 예측값 :  [[1.0740244e+04]
#  [6.9902945e+00]]
# 테스트 로스가 예측된 로스보다 좋다