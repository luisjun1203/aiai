from keras.datasets import fashion_mnist
from keras.models import Sequential, Model
import numpy as np
from keras.datasets import mnist
import pandas as pd
from keras.layers import Dense, Conv2D, Flatten, Dropout, GlobalAveragePooling2D
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder,StandardScaler, RobustScaler
import time
from keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

# print(X_train.shape, y_train.shape)         # (60000, 28, 28) (60000,)
# print(X_test.shape, y_test.shape)           # (10000, 28, 28) (10000,)

# print(np.unique(y_train, return_counts=True))
# # (array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=uint8), array([6000, 6000, 6000, 6000, 6000, 6000, 6000, 6000, 6000, 6000], dtype=int64))
# print(np.unique(y_test, return_counts=True))
# # (array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=uint8), array([1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000], dtype=int64))

# plt.imshow(X_train[1])
# plt.show()

# print(pd.value_counts(y_test))

X_train = X_train.reshape(-1, 28, 28, 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)

y_test = y_test.reshape(-1,1)
y_train = y_train.reshape(-1,1)

ohe = OneHotEncoder(sparse=False)
y_train = ohe.fit_transform(y_train)
y_test = ohe.fit_transform(y_test)

X_train = X_train/255           
X_test = X_test/255
# mean = np.mean(X_train, axis=(0, 1, 2))
# std = np.std(X_train, axis=(0, 1, 2))
# X_train = (X_train - mean) / std
# X_test = (X_test - mean) / std
# print(X_train)            # (60000, 28, 28, 1)
# print(X_test)             # (10000, 28, 28, 1)


model = Sequential()                    
model.add(Conv2D(19, kernel_size=(3, 3), input_shape=(28, 28, 1), activation='swish', strides=2, padding='same'))   
model.add(Conv2D(97, (4, 4), activation='swish', strides=2, padding='same'))                         
model.add(Conv2D(210, (3, 3), activation='swish', strides=2, padding='same'))              
model.add(GlobalAveragePooling2D())  
model.add(Dense(124, activation='swish'))
model.add(Dropout(0.3)) 
model.add(Dense(48, activation='swish'))
model.add(Dense(10, activation='softmax'))

# model.summary()

# 3. 컴파일, 훈련

strat_time = time.time()
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
es = EarlyStopping(monitor='val_loss', mode='min', patience=21, verbose=2, restore_best_weights=True)
model.fit(X_train, y_train, epochs=210, batch_size=21,verbose=2, validation_split=0.2, callbacks=[es])
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









