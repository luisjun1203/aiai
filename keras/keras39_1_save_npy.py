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


train_datagen = ImageDataGenerator(
    rescale=1./255,             # 크기 스케일링
    # horizontal_flip=True,       # 수평 뒤집기
    # vertical_flip=True,      # 수직 뒤집기
    # width_shift_range=0.1,      # 0.1만큼 평행이동
    # height_shift_range=0.1,     # 0.1만큼 수직이동
    # rotation_range=5,           # 정해진 각도만큼 이미지를 회전
    # zoom_range=1.2,             # 1.2배 확대(축소도 가능)
    # shear_range=0.7,            # 좌표 하나를 고정시키고 다른 몇 개의 좌표를 이동시키는 변환
    # fill_mode='nearest',        # 비어있는 데이터에 근처 가장 비슷한 값으로 변환(채워줌)                
)

test_datagen = ImageDataGenerator(rescale=1./255)      # 평가지표이기 때문에 건드리지마         

path_train = "c:\\_data\\image\\brain\\train\\"
path_test = "c:\\_data\\image\\brain\\test\\"

xy_train = train_datagen.flow_from_directory(
    path_train, 
    target_size=(150,150),              # 사이즈 조절
    batch_size=200,                      # 160이상을 쓰면 x 통데이터로 가져올 수 있다 , 몇 장씩 수치화 시킬건지 정해준다           
    class_mode='binary',
    color_mode= 'grayscale',
    shuffle=True)


    


xy_test = test_datagen.flow_from_directory(
    path_test, 
    target_size=(150,150),              # 사이즈 조절
    batch_size=200,   
    color_mode='grayscale',                  
    class_mode='binary')

# Found 160 images belonging to 2 classes.
# Found 120 images belonging to 2 classes.


# start_time = time.time()
# X = []
# y = []

# for i in range(len(xy_train)):
#     batch = xy_train.next()
#     X.append(batch[0])          # 현재 배치의 이미지 데이터
#     y.append(batch[1])          # 현재 배치의 라벨 데이터
# X = np.concatenate(X, axis=0)   # 리스트에 저장된 여러개의 NUMPY 배열들을 행을 따라 연결하여 하나의 큰 배열을 만들어줌
# y = np.concatenate(y, axis=0)   # 리스트에 저장된 여러개의 NUMPY 배열들을 행을 따라 연결하여 하나의 큰 배열을 만들어줌
# end_time = time.time()



# print("걸린시간 : ", round(end_time-start_time, 2),"초")


# print(xy_train)
# # Found 160 images belonging to 2 classes.

# print(xy_test)
# Found 120 images belonging to 2 classes.
# start_time = time.time()
# print(xy_train.next())


# print(xy_train[0][0])       # 첫번째 배치의 x
# print(xy_train[0][1])       # 첫번째 배치의 y
# print(xy_train[0][0].shape)       # (160, 150, 150, 1)
# print(xy_train[0][1].shape)       # (160,)
# print(xy_test[0][0].shape)       # (120, 150, 150, 1)
# print(xy_test[0][1].shape)        # (120,)

#### Numpy로 변환해서 저장하는법 ###########

np_path = "c:\\_data\\_save_npy\\"

np.save(np_path + 'keras39_1_X_train.npy', arr=xy_train[0][0])            # (160, 150, 150, 1) 이 데이터가   'keras39_1_X_train.npy 여기로 저장된다   
np.save(np_path + 'keras39_1_y_train.npy', arr=xy_train[0][1])           
np.save(np_path + 'keras39_1_X_test.npy', arr=xy_test[0][0])           
np.save(np_path + 'keras39_1_y_test.npy', arr=xy_test[0][1])           




'''
# 1. 데이터


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
# es = EarlyStopping(monitor='val_loss', mode='min', patience=30, verbose=2, restore_best_weights=True)

model.fit_generator(xy_train,                                 # fit_generator 곧 소멸될듯  소멸되면 그냥 fit안에 포함될듯
# model.fit(xy_train,           
                    steps_per_epoch=16,           # 전체 데이터 / batch = 160 / 10 = 16
                    epochs=100,
                    # batch_size=50,            # fit_generator에서는 에러, fit에서는 안먹힘   *위 batch_size에서 조절*
                    # verbose=2,
                    # validation_split=0.15,     # 에러, validation : 검증
                    # callbacks=[es]
                    validation_data=xy_test
                    )
# end_time = time.time()
#   UserWarning: `Model.fit_generator` is deprecated and will be removed in a future version. 
#   Please use `Model.fit`, which supports generators.


# 4. 평가, 예측
loss = model.evaluate(xy_test)

# y_predict = model.predict(X_test)
# y_predict = y_predict.round()

# print(y_predict)
# print("걸린시간 : ", round(end_time-start_time, 2),"초")

print("로스 : ", loss[0])
print("acc : ", loss[1])

'''



# 걸린시간 :  6.55 초
# 로스 :  0.026794923469424248
# acc :  1.0










