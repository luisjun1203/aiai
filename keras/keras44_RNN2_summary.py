import numpy as np
from keras.models import Sequential, Model
from keras.layers import Dense,Dropout, Conv2D,SimpleRNN
from keras.callbacks import EarlyStopping
# 1. 데이터

datasets = np.array([1,2,3,4,5,6,7,8,9,10])


X = np.array([[1,2,3],
             [2,3,4],
             [3,4,5],
             [4,5,6],
             [5,6,7],
             [6,7,8],
             [7,8,9],
             ])

y = np.array([4,5,6,7,8,9,10,])


print(X.shape)      # (7, 3)
print(y.shape)      # (7, )

X = X.reshape(-1, 3, 1)
print(X.shape)      # (7, 3, 1)

model = Sequential()
model.add(SimpleRNN(units = 10, input_shape = (3, 1), activation='tanh'))              # input_shape : 3-D tensor with shape (batch_size, timesteps, features)
# model.add(Dense(97,activation='relu'))                파라미터 수 : ((units X units) + (units X features) + (units X bias))
# model.add(Dense(9,activation='relu'))                                                                                                  
# model.add(Dense(21,activation='swish'))
# model.add(Dense(16,activation='relu'))
# model.add(Dense(21,activation='relu'))
model.add(Dense(7,activation='relu'))
model.add(Dense(1))

model.summary()

# _________________________________________________________________
#  Layer (type)                Output Shape              Param #
# =================================================================
#  simple_rnn (SimpleRNN)      (None, 10)                120

#  dense (Dense)               (None, 7)                 77

#  dense_1 (Dense)             (None, 1)                 8

# =================================================================
# Total params: 205 (820.00 Byte)
# Trainable params: 205 (820.00 Byte)
# Non-trainable params: 0 (0.00 Byte)

# 파라미터 갯수 = units X (units + bias + features)



model.compile(loss = 'mse', optimizer='adam')
model.fit(X, y, epochs=10000)

results = model.evaluate(X,y)
print("loss : ", results)
y_predict = np.array([8,9,10]).reshape(1, 3, 1)
y_predict = model.predict(y_predict)
print("[8,9,10]의 예측값 : ", y_predict)



    # [8,9,10]의 예측값 :  [[10.909407]]
    
    
    
    
    
    