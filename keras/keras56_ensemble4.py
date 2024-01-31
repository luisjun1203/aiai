import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Model, Sequential
from keras.layers import Dense, Input, concatenate, Concatenate



# 1. 데이터

X1_datasets = np.array([range(100), range(301, 401)]).T         # ex) 삼성전자 종가, 하이닉스 종가

# print(X1_datasets.shape, X2_datasets.shape) # (100, 2) (100, 3)

y1 = np.array(range(3001, 3101))             # 비트코인 종가
y2 = np.array(range(13001, 13101))             # 이더리움

X1_train, X1_test, y1_train, y1_test, y2_train, y2_test = train_test_split(
    X1_datasets, y1, y2, test_size=0.15, random_state=3)

# print(X1_train.shape)   #(85, 2)
# print(X2_train.shape)   #(85, 3)
# print(y_train.shape)    #(85,)

# 2-1. 모델구성1
i1 = Input(shape=(2,))
d1 = Dense(10, activation='relu', name= 'bit1')(i1)
d2 = Dense(10, activation='relu', name= 'bit2')(d1)
d3 = Dense(10, activation='relu', name= 'bit3')(d2)
o1 = Dense(10, activation='relu', name= 'bit4')(d3)


# # 2-2. 모델구성 2
# i2 = Input(shape=(3,))
# d11 = Dense(100, activation='relu', name= 'bit11')(i2)
# d12 = Dense(100, activation='relu', name= 'bit12')(d11)
# d13 = Dense(100, activation='relu', name= 'bit13')(d12)
# o2 = Dense(100, activation='relu', name= 'bit14')(d13)

# 2-3. 모델구성3


m1 = concatenate([o1], name = 'mg1')
m2 = Dense(7, name='mg2')(m1)
m3 = Dense(11, name='mg3')(m2)

final_layer = Dense(5, activation='relu')(m3)

fo = Dense(1,activation='relu', name= 'last')(final_layer)
fo1 = Dense(1,activation='relu',name='last2' )(final_layer)
model = Model(inputs=i1, outputs=[fo, fo1])

model.summary()

# 2-3. concatenate

model.compile(loss='mse', optimizer='adam')
model.fit(X1_train, [y1_train, y2_train] ,epochs=1500, batch_size=30)


results = model.evaluate(X1_test, [y1_test, y2_test])

print("로스 : ", results)

