import numpy as np
import pandas as pd
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout,Input, Conv2D, MaxPooling2D, Flatten, GlobalAveragePooling2D,concatenate, Reshape, LSTM
from keras. callbacks import EarlyStopping, ModelCheckpoint
from keras. utils import to_categorical
from sklearn.model_selection import train_test_split 
from sklearn.metrics import f1_score
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler


a = np.array(range(1, 101))
X_predict = np.array(range(96, 106))
size = 5        # X데이터는 4게, y데이터는 1개

def split_X(dataset, size):
    aaa=[]
    for i in range(len(dataset) - size + 1):
        subset = dataset[i : (i + size)]
        aaa.append(subset)
    return np.array(aaa)
    

bbb = split_X(a, size)
# print(bbb)          # [[ 1  2  3  4  5][ 2  3  4  5  6][ 3  4  5  6  7][ 4  5  6  7  8][ 5  6  7  8  9][ 6  7  8  9 10]]
# print(bbb.shape)    # (6, 5)

X = bbb[:, :-1]       
y = bbb[:, -1]
# print(X,y)                         

ccc = split_X(X_predict, size - 1)
# print(ccc)
# print(ccc)
print(ccc.shape)    # (7, 4)

# X = X.reshape(-1, 2, 2)
# ccc = ccc.reshape(-1, 2, 2)
print(X.shape, y.shape)         # (96, 4, 1) (96,)    


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=3)

model = Sequential()
model.add(Dense(19,          
               input_shape=(4, ), activation='relu'))         # (N, 4, 1) -> (N, 2, 2)도 사용 가능
model.add(Dense(97, ))                                                       # timesteps와 feature은 reshape 가능
model.add(Dense(9, activation='swish'))
model.add(Dense(21, activation='swish'))
model.add(Dense(16, activation='swish'))
model.add(Dense(21, activation='swish'))
model.add(Dense(1))
model.summary()

model.compile(loss='mse', optimizer='adam')
es = EarlyStopping(monitor='loss', mode=min, patience=300, verbose=1)
model.fit(X, y, epochs=2000, callbacks=[es])


results = model.evaluate(X,y)
print("loss : ", results) 
y_predict = model.predict(ccc)

print("[X_predict]의 예측값 : ", y_predict)

# loss :  0.00014649158401880413
# [X_predict]의 예측값 :  [[ 99.99512 ]
#  [100.99655 ]
#  [101.99801 ]
#  [102.999596]
#  [104.00116 ]
#  [105.00281 ]
#  [106.00451 ]]

