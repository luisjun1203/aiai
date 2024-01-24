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
    target_size=(100,100),              # 사이즈 조절
    batch_size=100,                      # 160이상을 쓰면 x 통데이터로 가져올 수 있다 , 몇 장씩 수치화 시킬건지 정해준다           
    class_mode='binary',
    shuffle=True)

X_train = []
y_train = []

for i in range(len(xy_train)):
    batch = xy_train.next()
    X_train.append(batch[0])          # 현재 배치의 이미지 데이터
    y_train.append(batch[1])          # 현재 배치의 라벨 데이터
X_train = np.concatenate(X_train, axis=0)   # 리스트에 저장된 여러개의 NUMPY 배열들을 행을 따라 연결하여 하나의 큰 배열을 만들어줌
y_train = np.concatenate(y_train, axis=0)   # 리스트에 저장된 여러개의 NUMPY 배열들을 행을 따라 연결하여 하나의 큰 배열을 만들어줌
    


xy_test = test_datagen.flow_from_directory(
    path_test, 
    target_size=(100,100),              # 사이즈 조절
    batch_size=100,                       
    class_mode='binary')

X_test=[]
y_test=[]

for i in range(len(xy_test)):
    batch = xy_test.next()
    X_test.append(batch[0])          # 현재 배치의 이미지 데이터
    y_test.append(batch[1])          # 현재 배치의 라벨 데이터
X_test = np.concatenate(X_test, axis=0)   # 리스트에 저장된 여러개의 NUMPY 배열들을 행을 따라 연결하여 하나의 큰 배열을 만들어줌
y_test = np.concatenate(y_test, axis=0)


# print(X_train.shape, y_train.shape)
# print(X_test.shape, y_test.shape)





# start_time = time.time()







# end_time = time.time()
# print(X.shape)      #(19996, 150, 150, 3)
# print(y.shape)      # (19996,)



np_path = "c:\\_data\\_save_npy\\"

np.save(np_path + 'keras37_3_X_train.npy', arr=X_train)              
np.save(np_path + 'keras37_3_y_train.npy', arr=y_train)           
np.save(np_path + 'keras37_3_X_test.npy', arr=X_test)           
np.save(np_path + 'keras37_3_y_test.npy', arr=y_test) 

