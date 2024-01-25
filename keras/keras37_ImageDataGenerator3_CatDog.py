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

path_train = "c:\\_data\\image\\cat_and_dog\\train\\"
path_test = "c:\\_data\\image\\cat_and_dog\\test\\"

xy_train = train_datagen.flow_from_directory(
    path_train, 
    target_size=(150,150),              # 사이즈 조절
    batch_size=500,                      # 160이상을 쓰면 x 통데이터로 가져올 수 있다 , 몇 장씩 수치화 시킬건지 정해준다           
    class_mode='binary',
    shuffle=True)


    


# xy_test = test_datagen.flow_from_directory(
#     path_test, 
#     target_size=(120,120),              # 사이즈 조절
#     batch_size=5000,                       
#     class_mode='binary')




# start_time = time.time()
X = []
y = []

for i in range(len(xy_train)):
    batch = xy_train.next()
    X.append(batch[0])          # 현재 배치의 이미지 데이터
    y.append(batch[1])          # 현재 배치의 라벨 데이터
X = np.concatenate(X, axis=0)   # 리스트에 저장된 여러개의 NUMPY 배열들을 행을 따라 연결하여 하나의 큰 배열을 만들어줌
y = np.concatenate(y, axis=0)   # 리스트에 저장된 여러개의 NUMPY 배열들을 행을 따라 연결하여 하나의 큰 배열을 만들어줌
# end_time = time.time()

np_path = "c:\\_data\\_save_npy\\"

np.save(np_path + 'keras37_1_X_train.npy', arr=X)            # (160, 150, 150, 1) 이 데이터가   'keras39_1_X_train.npy 여기로 저장된다   
np.save(np_path + 'keras37_1_y_train.npy', arr=y)           
# np.save(np_path + 'keras39_1_X_test.npy', arr=xy_test[0][0])           
# np.save(np_path + 'keras39_1_y_test.npy', arr=xy_test[0][1]) 




# print(X.shape)    #  (19996, 500, 500, 3)
# print(y.shape)    #  (19996,)
# print("걸린시간 : ", round(end_time-start_time, 2),"초")
# 걸린시간 :  304.67 초

# print(xy_train)
# # <keras.preprocessing.image.DirectoryIterator object at 0x00000242C4B8BB50>
# # Found 20000 images belonging to 2 classes.

# print(xy_test)
# Found 5000 images belonging to 2 classes.
# start_time = time.time()
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

# X_train = xy_train[0][0]
# y_train = xy_train[0][1]
# X_test = xy_test[0][0]
# y_test = xy_test[0][1]

X_train, X_test, y_train, y_test = train_test_split(X , y, test_size=0.15, shuffle=True, random_state=3, stratify=y)
# print(xy_train.next())



model = Sequential()
model.add(Conv2D(5, (3,3), activation='relu', input_shape=(150, 150, 3)))
model.add(MaxPooling2D())
model.add(Conv2D(7, (4,4), activation='relu'))
model.add(MaxPooling2D())
model.add(Conv2D(4, (3,3), activation='relu'))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(8,activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.summary()







start_time = time.time()
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
es = EarlyStopping(monitor='val_loss', mode='min', patience=70, verbose=2, restore_best_weights=True)
model.fit(X_train, y_train, epochs=300, batch_size=50, verbose=2, validation_split=0.15, callbacks=[es])
# end_time = time.time()
end_time = time.time()



loss = model.evaluate(X_test, y_test)

# y_predict = model.predict(X_test)
# y_predict = y_predict.round()

# print(y_predict)
print("걸린시간 : ", round(end_time - start_time, 2),"초")

print("로스 : ", loss[0])
print("acc : ", loss[1])





# gpu (150,150), batch 20
# 걸린시간 :  227.74 초
# 로스 :  0.5193547010421753
# acc :  0.7505500912666321



# cpu (300,300), batch 5
# 걸린시간 :  808.23 초
# 로스 :  0.5657410025596619
# acc :  0.7129426002502441

# gpu (150,150), batch 10
# 걸린시간 :  176.59 초
# 로스 :  0.5184306502342224
# acc :  0.744148850440979


# gpu (150,150), batch 30
# 걸린시간 :  134.65 초
# 로스 :  0.5588529706001282
# acc :  0.7271454334259033


# gpu (150,150), batch 15
# 걸린시간 :  223.66 초
# 로스 :  0.5664209723472595
# acc :  0.7229446172714233


# gpu batch 500
# 걸린시간 :  343.97 초
# 로스 :  0.47394585609436035
# acc :  0.7876666784286499



