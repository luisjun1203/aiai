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


# 1. 데이터

#### Numpy로 변환해서 저장하는법 ###########

np_path = "c:\\_data\\_save_npy\\"

# np.save(np_path + 'keras39_1_X_train.npy', arr=xy_train[0][0])            # (160, 150, 150, 1) 이 데이터가   'keras39_1_X_train.npy 여기로 저장된다   
# np.save(np_path + 'keras39_1_y_train.npy', arr=xy_train[0][1])           
# np.save(np_path + 'keras39_1_X_test.npy', arr=xy_test[0][0])           
# np.save(np_path + 'keras39_1_y_test.npy', arr=xy_test[0][1])           

X_train = np.load(np_path + 'keras39_1_X_train.npy')
y_train = np.load(np_path + 'keras39_1_y_train.npy')
X_test = np.load(np_path + 'keras39_1_X_test.npy')
y_test = np.load(np_path + 'keras39_1_y_test.npy')

print(X_train)
print(X_train.shape, y_train.shape)        # (160, 150, 150, 1) (160,)
print(X_test.shape, y_test.shape)          # (120, 150, 150, 1) (120,)








# 2. 모델구성
model = Sequential()
model.add(Conv2D(5, (3,3), activation='relu', input_shape=(150, 150, 1)))
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
es = EarlyStopping(monitor='val_loss', mode='min', patience=30, verbose=2, restore_best_weights=True)

model.fit(X_train, y_train,                                 # fit_generator 곧 소멸될듯  소멸되면 그냥 fit안에 포함될듯
# model.fit_generatr(xy_train,           
                    # steps_per_epoch=16,           # 전체 데이터 / batch = 160 / 10 = 16
                    epochs=100,
                    batch_size= 30,            # fit_generator에서는 에러, fit에서는 안먹힘   *위 batch_size에서 조절*
                    # verbose=2,
                    validation_split=0.15,     # 에러, validation : 검증
                    callbacks=[es]
                    # validation_data=xy_test
                    )
# end_time = time.time()


# 4. 평가, 예측
loss = model.evaluate(X_test, y_test)

# y_predict = model.predict(X_test)
# y_predict = y_predict.round()

# print(y_predict)
# print("걸린시간 : ", round(end_time-start_time, 2),"초")

print("로스 : ", loss[0])
print("acc : ", loss[1])





# 걸린시간 :  6.55 초
# 로스 :  0.026794923469424248
# acc :  1.0










