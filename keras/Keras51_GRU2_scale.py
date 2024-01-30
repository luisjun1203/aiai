import numpy as np
from keras.models import Sequential, Model
from keras.layers import Dense,Dropout, Conv2D,SimpleRNN, LSTM, Bidirectional, GRU
from keras.callbacks import EarlyStopping



# 1. 데이터
X = np.array([[1,2,3], [2,3,4], [3,4,5], [4,5,6],
              [5,6,7], [6,7,8], [7,8,9], [8,9,10],
              [9,10,11], [10,11,12],
              [20,30,40], [30,40,50], [40,50,60],
              ])
y = np.array([4,5,6,7,8,9,10,11,12,13,50,60,70])

X = X.reshape(-1, 3, 1)
print(X.shape) #(13,3,1)


model = Sequential()
model.add(Bidirectional(GRU(19, return_sequences=True, activation='relu'), input_shape=(3, 1)))
model.add(Bidirectional(GRU(97, )))
model.add(Dense(9, activation='swish'))
model.add(Dense(21, activation='swish'))
model.add(Dense(16, activation='swish'))
model.add(Dense(21, activation='swish'))
model.add(Dense(1))
model.summary()


model.compile(loss='mse', optimizer='adam')
es = EarlyStopping(monitor='loss', mode=min, patience=100, verbose=1)
model.fit(X, y, epochs=1000, callbacks=[es])


results = model.evaluate(X,y)
print("loss : ", results)

X_predict = np.array([50,60,70]).reshape(-1,3,1)
y_predict = model.predict(X_predict)
print("[50,60,70]의 예측값 : ", y_predict)


# [50,60,70]의 예측값 :  [[80.010216]]

# [50,60,70]의 예측값 :  [[77.6676]]

# [50,60,70]의 예측값 :  [[77.42349]]
