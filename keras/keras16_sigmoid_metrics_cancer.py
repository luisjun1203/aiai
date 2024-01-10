import numpy as np
from sklearn.datasets import load_breast_cancer
from keras.models import Sequential
from keras.layers import Dense
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error,mean_squared_log_error
from keras.callbacks import EarlyStopping


#1. 데이터
datasets= load_breast_cancer()
# print(datasets)
print(datasets.DESCR)
print(datasets.feature_names)

X = datasets.data
y = datasets.target

print(X.shape, y.shape)     # (569, 30) (569,)
print(np.unique(y, return_counts=True))         # [0 1], (array([0, 1]), array([212, 357], dtype=int64))
# print(y[np.where(y==0)].size)   #212
# print(y[np.where(y==1)].size)   #357

# print(pd.DataFrame(y).value_counts())           # 3가지 다 같다     #1    357 ,0    212
# print(pd.value_counts(y))
# print(pd.Series(y).value_counts())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=True, random_state=713)

#2. 모델구성
model = Sequential()
model.add(Dense(19, input_dim = 30, activation='relu')) # activation='sigmoid' 이진분류모델이 나오면 무주건 써야한다.
model.add(Dense(97,activation='relu'))
model.add(Dense(9))
model.add(Dense(21,activation='relu'))
model.add(Dense(99,activation='relu'))
model.add(Dense(7))
model.add(Dense(1, activation = 'sigmoid'))


#3.컴파일,훈련

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', 'mse', 'mae'])                             #이진 분류 모델이 나오면 "binary_crossentropy"                          #분류모델에서는 mse사용안함
es = EarlyStopping(monitor='val_loss', mode='min', patience=100, verbose=3, restore_best_weights=True)
hist = model.fit(X_train, y_train, epochs=300, batch_size=50, validation_split=0.15, callbacks=[es], )


#4.평가,예측

loss = model.evaluate(X_test, y_test)
y_predict = model.predict(X_test)
r2 = r2_score(y_test, y_predict)
result = model.predict(X)
print("로스 : ", loss)
print("R2 : ", r2)
def RMSE(aaa, bbb):
    np.sqrt(mean_squared_error(aaa, bbb))
    return np.sqrt(mean_squared_error(aaa, bbb))
rmse = RMSE(y_test, y_predict)

print("RMSE : ", rmse)
# print("???",result)


import matplotlib.pyplot as plt
plt.rc('font', family= 'Malgun Gothic')
plt.figure(figsize=(10, 5))
plt.plot(hist.history['loss'], color='red', label='loss',marker='.')
plt.plot(hist.history['val_loss'], c='blue', label='val_loss',marker='.')
plt.legend(loc = 'upper right')      # loc = location
plt.title('암 loss')
plt.xlabel('에포')
plt.ylabel('로스')
plt.grid()          
plt.show()
















