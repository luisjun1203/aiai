import numpy as np
import pandas as pd
from keras.preprocessing.image import ImageDataGenerator        ######
from keras.models import Sequential, Model
from keras.layers import Dense, Input, Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dropout, Flatten
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, LabelEncoder, OneHotEncoder
import time
import matplotlib.pyplot as plt

train_datagen = ImageDataGenerator(
    rescale=1./255,             # 크기 스케일링 , .의 의미는 부동소수점으로 인식을 해줘서 연산속도 살짝 빨라질 수 있음
    horizontal_flip=True,       # 수평 뒤집기
    vertical_flip=True,      # 수직 뒤집기
    width_shift_range=0.1,      # 0.1만큼 평행이동
    height_shift_range=0.1,     # 0.1만큼 수직이동
    rotation_range=30,           # 정해진 각도만큼 이미지를 회전
    zoom_range=[0.5, 0.9],             # 1.2배 확대(축소도 가능)
    shear_range=45,            # 좌표 하나를 고정시키고 다른 몇 개의 좌표를 이동시키는 변환
    fill_mode='nearest',        # 비어있는 데이터에 근처 가장 비슷한 값으로 변환(채워줌)      
    
              
)

test_datagen = ImageDataGenerator(rescale=1./255)      # 평가지표이기 때문에 건드리지마

path_train = "c:\\_data\\image\\brain\\train\\"
path_test = "c:\\_data\\image\\brain\\test\\"

xy_train = train_datagen.flow_from_directory(
    path_train, 
    target_size=(150,150),              # 사이즈 조절
    batch_size=160,                      # 160이상을 쓰면 x 통데이터로 가져올 수 있다            
    class_mode='binary',
    shuffle=True)



xy_test = test_datagen.flow_from_directory(
    path_test, 
    target_size=(150,150),              # 사이즈 조절
    batch_size=120,                       
    class_mode='binary')


X_train = xy_train[0][0]
y_train = xy_train[0][1]
X_test = xy_test[0][0]
y_test = xy_test[0][1]
# print(X_train.shape)       # (160, 200, 200, 3)
# print(y_train.shape)       # (160,)
# print(X_test.shape)        # (120, 200, 200, 3)
# print(y_test.shape)        # (120,)




model = Sequential()
model.add(Conv2D(5, (3,3), activation='swish', input_shape=(150, 150, 3)))
model.add(MaxPooling2D())
model.add(Conv2D(7, (4,4), activation='swish'))
model.add(MaxPooling2D())
model.add(Conv2D(4, (3,3), activation='swish'))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(8,activation='swish'))
model.add(Dense(1, activation='sigmoid'))

model.summary()

# strat_time = time.time()
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
es = EarlyStopping(monitor='val_loss', mode='min', patience=50, verbose=2, restore_best_weights=True)
model.fit(X_train, y_train, epochs=200, batch_size=10,verbose=2, validation_split=0.25, callbacks=[es])
# end_time = time.time()


loss = model.evaluate(X_test, y_test)

y_predict = model.predict(X_test)
y_predict = y_predict.round()

print(y_predict)
print("로스 : ", loss[0])
print("acc : ", loss[1])

# 로스 :  0.18828198313713074
# acc :  0.9416666626930237


