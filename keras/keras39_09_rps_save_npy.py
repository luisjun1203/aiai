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


train_datagen = ImageDataGenerator( rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

path_train = "c:\\_data\\image\\rps\\"

xy_train = train_datagen.flow_from_directory(
    path_train,
    target_size=(150,150),
    batch_size=50,
    class_mode= 'categorical',
    shuffle=True    
)

X = []
y = []

for i in range(len(xy_train)):
    batch = xy_train.next()
    X.append(batch[0])          
    y.append(batch[1])          
X = np.concatenate(X, axis=0)   
y = np.concatenate(y, axis=0)

print(X.shape)          # (2520, 150, 150, 3)
print(y.shape)          # (2520, 3)


np_path = "c:\\_data\\_save_npy\\"

np.save(np_path + 'keras39_9_X_train', arr = X)
np.save(np_path + 'keras39_9_y_train', arr = y)

print(X.shape)  # (2520, 150, 150, 3)
print(y.shape)  # (2520, 3)








