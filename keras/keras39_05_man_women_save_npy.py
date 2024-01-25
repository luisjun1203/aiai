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

train1_datagen = ImageDataGenerator(
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
# train2_datagen = ImageDataGenerator(
#     rescale=1./255,             # 크기 스케일링
#     horizontal_flip=True,       # 수평 뒤집기
#     vertical_flip=True,      # 수직 뒤집기
#     width_shift_range=0.1,      # 0.1만큼 평행이동
#     height_shift_range=0.1,     # 0.1만큼 수직이동
#     rotation_range=5,           # 정해진 각도만큼 이미지를 회전
#     zoom_range=1.2,             # 1.2배 확대(축소도 가능)
#     shear_range=0.7,            # 좌표 하나를 고정시키고 다른 몇 개의 좌표를 이동시키는 변환
#     fill_mode='nearest',        # 비어있는 데이터에 근처 가장 비슷한 값으로 변환(채워줌)  
                  
# )



# # train3_datagen = ImageDataGenerator(
# #     rescale=1./255,             # 크기 스케일링
# #     horizontal_flip=True,       # 수평 뒤집기
# #     vertical_flip=True,      # 수직 뒤집기
# #     width_shift_range=0.1,      # 0.1만큼 평행이동
# #     height_shift_range=0.1,     # 0.1만큼 수직이동
# #     rotation_range=5,           # 정해진 각도만큼 이미지를 회전
# #     zoom_range=1.2,             # 1.2배 확대(축소도 가능)
# #     shear_range=0.7,            # 좌표 하나를 고정시키고 다른 몇 개의 좌표를 이동시키는 변환
# #     fill_mode='nearest',        # 비어있는 데이터에 근처 가장 비슷한 값으로 변환(채워줌)  
                  
# # )

test_datagen = ImageDataGenerator(rescale=1./255)     

path_train = "c:\\_data\\kaggle\\men-women\\data\\"
path_men = "c:\\_data\\kaggle\\\\men-women\\men\\"
path_women = "c:\\_data\\kaggle\\\\men-women\\women\\"

xy_train_data = train1_datagen.flow_from_directory(
    path_train, 
    target_size=(300,300),             
    batch_size=3309,                                  
    class_mode='binary',
    # color_mode= 'rgb',           
    shuffle=True)

# xy_train_men = train2_datagen.flow_from_directory(
#     path_men, 
#     target_size=(200,200),             
#     batch_size=1409,                                  
#     class_mode='binary',
#     # color_mode= 'rgb',           
    
#     shuffle=True)

xy_test_women = test_datagen.flow_from_directory(
    path_women, 
    target_size=(300,300),             
    batch_size=1900,                                  
    class_mode='binary',
    # color_mode= 'rgb',           
    
    shuffle=True)
# # Found 3309 images belonging to 2 classes.
# # Found 1409 images belonging to 1 classes.
# # Found 1900 images belonging to 1 classes

# print(xy_train_data)
# print(xy_train_men)
# print(xy_test_women)
print(xy_train_data[0][0].shape)    # (3309, 300, 300, 3)

print(xy_train_data[0][1].shape)    # (3309,)

print(xy_test_women[0][0].shape)    # (1900, 300, 300, 3)
np_path = "c:\\_data\\_save_npy\\"

np.save(np_path + 'keras39_5_X_train.npy', arr=xy_train_data[0][0])            # (160, 150, 150, 1) 이 데이터가   'keras39_1_X_train.npy 여기로 저장된다   
np.save(np_path + 'keras39_5_y_train.npy', arr=xy_train_data[0][1])           
np.save(np_path + 'keras39_5_X_test.npy', arr=xy_test_women[0][0])           
# np.save(np_path + 'keras39_5_X_train_men.npy', arr=xy_train_men[0][0])           
# np.save(np_path + 'keras39_5_X_train_men.npy', arr=xy_train_men[0][0])           






