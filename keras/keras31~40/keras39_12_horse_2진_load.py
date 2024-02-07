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



np_path = "c:\\_data\\_save_npy\\"

X = np.load(np_path + 'keras39_11_X_train.npy')
y = np.load(np_path + 'keras39_11_y_train.npy')


print(X.shape)      # (1027, 300, 300, 3)
print(y.shape)      # (1027, )



X_train, X_test, y_train, y_test = train_test_split(X , y, test_size=0.2, shuffle=True, random_state=3, stratify=y)





model = Sequential()
model.add(Conv2D(9, (4,4), activation='relu', input_shape=(300, 300, 3)))
model.add(MaxPooling2D())
model.add(Conv2D(5, (2,2), activation='relu'))
model.add(MaxPooling2D())
model.add(Conv2D(7, (3,3), activation='relu'))
model.add(MaxPooling2D())
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(4,activation='swish'))
model.add(Dense(2, activation='softmax'))
model.summary()

strat_time = time.time()
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
es = EarlyStopping(monitor='val_loss', mode='min', patience=15, verbose=2, restore_best_weights=True)
model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=2, callbacks=[es], validation_split=0.15)
end_time = time.time()

loss = model.evaluate(X_test, y_test)
print("loss : ", loss[0])
print("acc : ", loss[1])
print("걸린시간 : ", round(end_time - strat_time, 3), "초")

