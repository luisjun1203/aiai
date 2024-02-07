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

def save_code_to_file(filename=None):
    if filename is None:
        # 현재 스크립트의 파일명을 가져와서 확장자를 txt로 변경
        filename = os.path.splitext(os.path.basename(__file__))[0] + ".txt"
    else:
        filename = filename + ".txt"
    with open(__file__, "r") as file:
        code = file.read()
    
    with open(filename, "w") as file:
        file.write(code)

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

path_train = "c:\\_data\\image\\horse_human\\"
path_test = "c:\\_data\\image\\horse_human\\"

xy_train = train_datagen.flow_from_directory(
    path_train, 
    target_size=(300,300),              # 사이즈 조절
    batch_size=1027,                      # 160이상을 쓰면 x 통데이터로 가져올 수 있다 , 몇 장씩 수치화 시킬건지 정해준다           
    class_mode='categorical',
    # color_mode= 'grayscale',
    shuffle=True)




print(xy_train[0][0].shape)       # (1027, 300, 300, 3)
print(xy_train[0][1].shape)       # (1027, 2)


np_path = "c:\\_data\\_save_npy\\"

np.save(np_path + 'keras39_7_X_train', arr=xy_train[0][0])
np.save(np_path + 'keras39_7_y_train', arr=xy_train[0][1])



