import numpy as np
import pandas as pd
from keras.preprocessing.image import ImageDataGenerator       ######
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
    rescale=1./255,             # 크기 스케일링
    horizontal_flip=True,       # 수평 뒤집기
    vertical_flip=True,      # 수직 뒤집기
    width_shift_range=0.1,      # 0.1만큼 평행이동
    height_shift_range=0.1,     # 0.1만큼 수직이동
    rotation_range=5,           # 정해진 각도만큼 이미지를 회전
    zoom_range=1.2,             # 1.2배 확대(축소도 가능)
    shear_range=0.7,            # 좌표 하나를 고정시키고 다른 몇 개의 좌표를 이동시키는 변환
    fill_mode='nearest',        # 비어있는 데이터에 근처 가장 비슷한 값으로 변환(채워줌)                
)

test_datagen = ImageDataGenerator(rescale=1./255)      # 평가지표이기 때문에 건드리지마

path_train = "c:\\_data\\image\\brain\\train\\"
path_test = "c:\\_data\\image\\brain\\test\\"

xy_train = train_datagen.flow_from_directory(
    path_train, 
    target_size=(200,200),              # 사이즈 조절
    batch_size=50,                      # 160이상을 쓰면 x 통데이터로 가져올 수 있다            
    class_mode='binary',
    shuffle=True)



xy_test = test_datagen.flow_from_directory(
    path_test, 
    target_size=(200,200),              # 사이즈 조절
    batch_size=50,                       
    class_mode='binary')

# print(xy_train)
# # <keras.preprocessing.image.DirectoryIterator object at 0x00000242C4B8BB50>
# # Found 160 images belonging to 2 classes.

# print(xy_test)
# Found 120 images belonging to 2 classes.

# print(xy_train.next())
# print(xy_train[0])      # array([1., 0., 1., 1., 1., 0., 1., 0., 0., 0.], dtype=float32)) : y값
# print(xy_train[9])      # array([1., 0., 1., 1., 0., 0., 1., 0., 0., 0.], dtype=float32))
# print(xy_train[16])     # error : 전체 데이터/batch_size = 160/10 =16개인데 [16]은 17번째 값이라 에러가 나온다

# print(xy_train[0][0])       # 첫번째 배치의 x
# print(xy_train[0][1])       # 첫번째 배치의 y
# print(xy_train[0][0].shape)       # (10, 200, 200, 3)
# print(xy_train[0][1].shape)       # (10, )

# print(type(xy_train))               # <class 'keras.src.preprocessing.image.DirectoryIterator'>
# print(type(xy_train[0]))            # <class 'tuple'>
# print(type(xy_train[0][0]))         # <class 'numpy.ndarray'>
# print(type(xy_train[0][1]))         # <class 'numpy.ndarray'>





model = Sequential()
model.add(Conv2D(5, (3,3), activation='swish', input_shape=(200, 200, 3)))
model.add(MaxPooling2D())
model.add(Conv2D(7, (4,4), activation='swish'))
model.add(MaxPooling2D())
model.add(Conv2D(4, (3,3), activation='swish'))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(8,activation='swish'))
model.add(Dense(1, activation='sigmoid'))

model.summary()


es = EarlyStopping(monitor='val_loss', mode='min', patience=50, verbose=2, restore_best_weights=True)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

model.fit(
    xy_train,
    steps_per_epoch=5,
    epochs = 5,
    validation_data=xy_test,
    validation_steps=5, 
    callbacks=[es])

loss = model.evaluate(xy_test, steps=5)

y_predict = model.predict(xy_test)
y_predict = y_predict.round()

print(y_predict)
print("로스 : ", loss[0])
print("acc : ", loss[1])