from keras.datasets import fashion_mnist
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt
from keras.utils.image_utils import img_to_array, load_img
import pandas as pd
from keras.models import Sequential, Model
from keras.layers import Dense, Input, Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dropout, Flatten
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, LabelEncoder, OneHotEncoder
import time
import matplotlib.pyplot as plt


(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

X_train = X_train/255.
X_test = X_test/255.

train_datagen = ImageDataGenerator(
    # rescale=1./255,
    horizontal_flip=True,
    vertical_flip=True,
    width_shift_range=0.2,
    height_shift_range=0.2,
    rotation_range=30,
    zoom_range=[0.8, 1.2],
    shear_range=30,
    fill_mode='nearest'
    
)
print(X_train.shape[0])

augment_size = 40000

randidx = np.random.randint(X_train.shape[0], size= augment_size)       #랜덤int값을 뽑는다
        # np.random.randint(60000, 40000) -> 6만개중에 4만개를 임의로 뽑아라
# print(randidx)          ## [54132 43738 45650 ... 35177  1138   697]
# print(randidx.shape)     # (40000,)
# print(np.min(randidx), np.max(randidx))     # 2 59993

X_augmented = X_train[randidx].copy()               # X_train이 변형될 확률이 있어서 안전빵으로 .copy해줌 
y_augmented = y_train[randidx].copy()               

# print(X_augmented)
# print(X_augmented.shape)        # (40000, 28, 28)
# print(y_augmented)
# print(y_augmented.shape)        # (40000,)
X_augmented = X_augmented.reshape(
    X_augmented.shape[0],
    X_augmented.shape[1],
    X_augmented.shape[2], 1)


X_augmented = train_datagen.flow(
    X_augmented,y_augmented,
    batch_size= augment_size,
    shuffle=False       # 이미 위에서 섞여서 셔플 x
)#.next()[0]
print(type(X_augmented))
print(X_augmented)
print(X_augmented.shape)    # (40000, 28, 28, 1)


X_train = X_train.reshape(60000, 28, 28, 1)
X_test = X_test.reshape(10000, 28, 28, 1)

X_train = np.concatenate((X_train, X_augmented))            # concatenate: 사슬처럼 엮어주다
y_train = np.concatenate((y_train, y_augmented))
print(X_train.shape, y_train.shape)             # (100000, 28, 28, 1) (100000,)
print(y_train.shape, y_test.shape)
print(np.unique(y_test, return_counts=True))











































