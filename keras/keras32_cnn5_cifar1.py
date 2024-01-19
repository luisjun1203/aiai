from keras.datasets import cifar10
import numpy as np
from keras.datasets import mnist
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder
import time
from keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score
# acc = 0.77이상

(X_train, y_train), (X_test, y_test) = cifar10.load_data()

# print(X_test.shape, y_test.shape)       #(10000, 32, 32, 3) (10000, 1)
# print(X_train.shape, y_train.shape)     #(50000, 32, 32, 3) (50000, 1)

# print(np.unique(y_train,return_counts=True))
# (array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=uint8), array([5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000], dtype=int64))
# print(pd.value_counts(y_test))
# X_train = X_train.astype('float32')
# X_test = X_train.astype('float32')


# print(X_test)
# print(X_train)


y_train = y_train.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)

ohe = OneHotEncoder(sparse=False)
y_train = ohe.fit_transform(y_train)
y_test = ohe.fit_transform(y_test)

# print(y_test)
# print(y_train)


model = Sequential()                    
model.add(Conv2D(19, kernel_size=(2, 2),input_shape = (32, 32, 3)))   
model.add(Conv2D(97, (3, 3 )))                         
model.add(Conv2D(9, (4, 4 )))              
model.add(Flatten())                                                  
model.add(Dense(21,activation='relu'))
model.add(Dense(17,activation='relu'))
model.add(Dense(30,activation='relu'))
model.add(Dense(19,activation='relu'))
model.add(Dense(10, activation= 'softmax'))



strat_time = time.time()
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
es = EarlyStopping(monitor='acc', mode='max', patience=100, verbose=20, restore_best_weights=True)
model.fit(X_train, y_train, epochs=1000, batch_size=500,verbose=2, validation_split=0.15, callbacks=[es])
end_time = time.time()
# print(X_train, X_test)

# 4. 평가, 예측

results = model.evaluate(X_test, y_test)
y_predict = model.predict(X_test)
y_test = np.argmax(y_test, axis=1)
y_predict = np.argmax(y_predict, axis=1)
acc = accuracy_score(y_test, y_predict)
# print(y_test)

print('loss' , results[0])
print('acc', results[1])
print("걸리시간 : ", round(end_time - strat_time, 3), "초")
print("accuracy_score : ", acc)



















