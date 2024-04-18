from keras.datasets import cifar100
import numpy as np
from keras.datasets import mnist
import pandas as pd
from keras.models import Sequential     
from keras.layers import Dense, Conv2D, Flatten, Dropout, GlobalAveragePooling2D, MaxPooling2D
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder,StandardScaler, RobustScaler
import time
from keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

(X_train, y_train), (X_test, y_test) = cifar100.load_data()

# print(X_train.shape, y_train.shape)     # (50000, 32, 32, 3) (50000, 1)
# print(X_test.shape, y_test.shape)       # (10000, 32, 32, 3) (10000, 1)

      
# plt.imshow(X_train[39], 'gray')
# plt.show()

# print(np.unique(y_train, return_counts=True))

# (array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,
#        17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33,
#        34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50,
#        51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67,
#        68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84,
#        85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99]), array([500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500,
#        500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500,
#        500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500,
#        500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500,
#        500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500,
#        500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500,
#        500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500,
#        500, 500, 500, 500, 500, 500, 500, 500, 500], dtype=int64))


ohe = OneHotEncoder(sparse=False)
y_train = ohe.fit_transform(y_train)
y_test = ohe.fit_transform(y_test)

##### standardScaler #######
# mean = np.mean(X_train, axis=(0, 1, 2))     
# std = np.std(X_train, axis=(0, 1, 2))       
# X_train = (X_train - mean) / std            
# X_test = (X_test - mean) / std


########## minmax scaler #########
X_train = X_train/255           
X_test = X_test/255

# rbs = RobustScaler()
# X_train = rbs.fit_transform(X_train.reshape(-1, 32*32*3)).reshape(-1, 32, 32, 3)
# X_test = rbs.fit_transform(X_train.reshape(-1, 32*32*3)).reshape(-1, 32, 32, 3)

print(X_train)    
print(X_test)      

# 2. 모델 구성

model = Sequential()                    
model.add(Conv2D(19, kernel_size=(3, 3), input_shape=(32, 32, 3), activation='swish', strides=2, padding='same'))   
model.add(Conv2D(97, (4, 4), activation='swish',strides=2, padding='same'))                         
model.add(Conv2D(210, (3, 3), activation='swish', strides=2, padding='same'))              
model.add(MaxPooling2D())  
# model.add(Flatten())
model.add(GlobalAveragePooling2D())
model.add(Dense(124, activation='swish'))
model.add(Dropout(0.3)) 
model.add(Dense(48, activation='swish'))
model.add(Dense(100, activation='softmax'))

# model.summary()

# 3. 컴파일, 훈련

strat_time = time.time()
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
es = EarlyStopping(monitor='val_loss', mode='min', patience=30, verbose=2, restore_best_weights=True)
model.fit(X_train, y_train, epochs=200, batch_size=21,verbose=2, validation_split=0.15, callbacks=[es])
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


# loss 2.0690841674804688
# acc 0.4645000100135803
# 걸리시간 :  367.938 초
# accuracy_score :  0.4645


# loss 2.0741589069366455
# acc 0.46160000562667847
# 걸리시간 :  481.229 초
# accuracy_score :  0.4616

# loss 2.03399658203125
# acc 0.4690000116825104
# 걸리시간 :  670.962 초
# accuracy_score :  0.469


# loss 2.0134806632995605
# acc 0.47040000557899475
# 걸리시간 :  813.738 초
# accuracy_score :  0.4704
