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

# xy_train = train_datagen.flow_from_directory(
#     path_train, 
#     target_size=(100,100),              # 사이즈 조절
#     batch_size=100,                      # 160이상을 쓰면 x 통데이터로 가져올 수 있다 , 몇 장씩 수치화 시킬건지 정해준다           
#     class_mode='binary',
#     shuffle=True)



np_path = "c:\\_data\\_save_npy\\"

# np.save(np_path + 'keras37_1_X_train.npy', arr=X)            # (160, 150, 150, 1) 이 데이터가   'keras39_1_X_train.npy 여기로 저장된다   
# np.save(np_path + 'keras37_1_y_train.npy', arr=y)  


X_train = np.load(np_path + 'keras37_3_X_train.npy')
y_train = np.load(np_path + 'keras37_3_y_train.npy')
X_test = np.load(np_path + 'keras37_3_X_test.npy')
y_test = np.load(np_path + 'keras37_3_y_test.npy')

# print(X.shape)      # (19996, 150, 150, 3)
# print(y.shape)      # (19996,)
# X_train, X_test, y_train, y_test = train_test_split(X , y, test_size=0.15, shuffle=True, random_state=3, stratify=y)



# 2. 모델 구성

model = Sequential()
model.add(Conv2D(5, (3,3), activation='relu', input_shape=(100, 100, 3)))
model.add(MaxPooling2D())
model.add(Conv2D(7, (4,4), activation='relu'))
model.add(MaxPooling2D())
model.add(Conv2D(4, (3,3), activation='relu'))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(8,activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.summary()

# 3. 컴파일, 훈련

# strat_time = time.time()
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
es = EarlyStopping(monitor='val_loss', mode='min', patience=10, verbose=2, restore_best_weights=True)
model.fit(X_train, y_train, epochs=20, batch_size=10, verbose=2, validation_split=0.15, callbacks=[es])
# end_time = time.time()
# end_time = time.time()

# 4. 평가, 훈련
loss = model.evaluate(X_test, y_test)

y_predict = model.predict(X_test)
y_predict = y_predict.round()

print(y_predict)
print(y_predict.shape)



# print("걸린시간 : ", round(end_time-start_time, 2),"초")

print("로스 : ", loss[0])
print("acc : ", loss[1])





















