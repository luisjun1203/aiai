# 14_overfit1 카피

import warnings
warnings.filterwarnings('ignore')

from sklearn.datasets import load_boston
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
from sklearn.model_selection import train_test_split
import time
import numpy as np

datasets = load_boston()
# print(datasets)

X = datasets.data
y = datasets.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, shuffle=True, random_state=20)

model = Sequential()
model.add(Dense(33,input_dim=13))
model.add(Dense(6))
model.add(Dense(21))
model.add(Dense(10))
model.add(Dense(21))
model.add(Dense(1))

model.compile(loss='mae', optimizer='adam')

from keras.callbacks import EarlyStopping                                                               # early stopping 개념, min,max, auto
es = EarlyStopping(monitor='loss', mode='max', 
                   patience=100, verbose= 1)                                # 'local min' , 'global min'

start_time = time.time()
hist = model.fit(X_train, y_train, epochs=1000, batch_size=15, validation_split=0.2, callbacks=[es])
end_time = time.time()

loss = model.evaluate(X_test, y_test)
print("로스 : ", loss)

y_predict = model.predict(X_test)
result = model.predict(X)

from sklearn.metrics import r2_score, mean_squared_error
r2 = r2_score(y_test, y_predict)
print("R2스코어 : ", r2)

def RMSE(aaa, bbb):
    np.sqrt(mean_squared_error(aaa,bbb))
    return np.sqrt(mean_squared_error(aaa,bbb))
rmse = RMSE(y_test, y_predict)

print("걸린 시간  : ", round(end_time - start_time, 2), "초")
print("RMSE : ", rmse)

print("=================== hist =======================")
print(hist)

print("================================================")
print(hist.history)     # 과제, 리스트[2개 이상], 딕셔너리{'key' : [value]}, 튜플 공부해오기!!

print("=================== loss =======================")           # value : 데이터 값
print(hist.history['loss'])

print("================ val_loss ======================")
print(hist.history['val_loss'])

print("================================================")

import matplotlib.pyplot as plt
plt.figure(figsize=(10,60))                 #  그래프 사이즈
plt.plot(hist.history['loss'], color = 'gold', label='loss', marker='.')
plt.plot(hist.history['val_loss'], c = 'pink', label='val_loss', marker='.')
plt.legend(loc = 'upper right')      # loc = location
plt.title('boston loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.grid()          # 격자
plt.show()


# plt.figure(figsize=(9,6))
# plt.scatter(hist.history['val_loss'])
# plt.show()











