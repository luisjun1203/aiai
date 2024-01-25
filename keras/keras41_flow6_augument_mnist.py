from keras.datasets import mnist
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


(X_train, y_train), (X_test, y_test) = mnist.load_data()

# X_train = X_train/255.
# X_test = X_test/255.

train_datagen = ImageDataGenerator(
    
    horizontal_flip=True,
    vertical_flip=True,
    width_shift_range=0.2,
    height_shift_range=0.2,
    rotation_range=30,
    zoom_range=[0.8, 1.2],
    shear_range=30,
    fill_mode='nearest'
    
)
# print(X_train.shape[0])

augment_size = 10000

randidx = np.random.randint(X_train.shape[0], size= augment_size)       


X_augmented = X_train[randidx].copy()               
y_augmented = y_train[randidx].copy()               
# print(X_augmented.shape, y_augmented.shape)     # (10000, 28, 28) (10000,)

X_augmented = X_augmented.reshape(
    X_augmented.shape[0],
    X_augmented.shape[1],
    X_augmented.shape[2], 1)

# print(X_augmented.shape)            # (10000, 28, 28, 1)
X_augmented = train_datagen.flow(
    X_augmented,y_augmented,
    batch_size= augment_size,
    shuffle=False       
).next()[0]
# print(type(X_augmented))
print(X_augmented)
print(X_augmented.shape)    # (10000, 28, 28, 1)


X_train = X_train.reshape(60000, 28, 28, 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)


X_train = np.concatenate((X_train, X_augmented))            
y_train = np.concatenate((y_train, y_augmented))
# print(X_train.shape, y_train.shape)             
# print(y_train.shape, y_test.shape)
# print(np.unique(y_test, return_counts=True))

y_test = y_test.reshape(-1,1)
y_train = y_train.reshape(-1,1)

ohe = OneHotEncoder(sparse=False)
y_train = ohe.fit_transform(y_train)
y_test = ohe.fit_transform(y_test)

# print(y_test)
# print(y_train)




model = Sequential()                    
model.add(Conv2D(19, kernel_size=(3, 3), input_shape=(28, 28, 1), activation='swish', strides=2, padding='same'))   
model.add(Conv2D(97, (4, 4), activation='swish', strides=2, padding='same'))                         
model.add(Conv2D(210, (3, 3), activation='swish', strides=2, padding='same'))              
model.add(GlobalAveragePooling2D())  
model.add(Dense(124, activation='swish'))
model.add(Dropout(0.3)) 
model.add(Dense(48, activation='swish'))
model.add(Dense(10, activation='softmax'))

# model.summary()

# 3. 컴파일, 훈련

strat_time = time.time()
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
es = EarlyStopping(monitor='val_loss', mode='min', patience=21, verbose=2, restore_best_weights=True)
model.fit(X_train, y_train, epochs=100, batch_size=5000,verbose=2, validation_split=0.2, callbacks=[es])
end_time = time.time()
# print(X_train, X_test)

# 4. 평가, 예측

results = model.evaluate(X_test, y_test)
y_predict = model.predict(X_test)
y_test = np.argmax(y_test, axis=1)
y_predict = np.argmax(y_predict, axis=1)
acc = accuracy_score(y_test, y_predict)
# print(y_test)

print('loss' , results[0])
print('acc', results[1])
print("걸리시간 : ", round(end_time - strat_time, 3), "초")
print("accuracy_score : ", acc)

# loss 0.3082456886768341
# acc 0.9133999943733215
# 걸리시간 :  15.39 초
# accuracy_score :  0.9134



