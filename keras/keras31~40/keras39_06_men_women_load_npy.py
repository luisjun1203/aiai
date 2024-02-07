import numpy as np
import pandas as pd
from keras.preprocessing.image import ImageDataGenerator       ###### 이미지를 숫자로 바꿔준다##########
from keras.models import Sequential, Model
from keras.layers import Dense, Input, Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dropout, Flatten
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, LabelEncoder, OneHotEncoder
import time
import matplotlib.pyplot as plt
import os


np_path = "c:\\_data\\_save_npy\\"


X = np.load(np_path + 'keras39_5_X_train.npy')
y = np.load(np_path + 'keras39_5_y_train.npy')
test = np.load(np_path + 'keras39_5_X_test.npy')

print(X.shape)      # (3309, 300, 300, 3)
print(y.shape)      # (3309,)
print(test.shape)   # (1900, 300, 300, 3)

X_train, X_test, y_train, y_test = train_test_split(X , y, test_size=0.25, shuffle=True, random_state=713, stratify=y)
# 2. 모델구성
model = Sequential()
model.add(Conv2D(19, (3,3), activation='relu', input_shape=(300, 300, 3)))
model.add(MaxPooling2D())
model.add(Conv2D(97, (4,4), activation='relu'))
model.add(MaxPooling2D())
model.add(Conv2D(9, (3,3), activation='relu'))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(21,activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.summary()



# 3. 컴파일, 훈련

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
es = EarlyStopping(monitor='val_loss', mode='min', patience=70, verbose=2, restore_best_weights=True)
model.fit(X_train, y_train, epochs=100, batch_size=20, verbose=2, callbacks=[es], validation_split=0.15)



# 4. 평가, 훈련
loss = model.evaluate(X_test, y_test)

y_predict = model.predict(test)
y_predict = y_predict.round()



print(y_predict)
print(y_predict.shape)

print("로스 : ", loss[0])
print("acc : ", loss[1])



 