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

X = np.load(np_path + 'keras39_7_X_train.npy')
y = np.load(np_path + 'keras39_7_y_train.npy')

train_datagen = ImageDataGenerator(
    # rescale=1./255,             # 크기 스케일링
    horizontal_flip=True,       # 수평 뒤집기
    vertical_flip=True,      # 수직 뒤집기
    width_shift_range=0.1,      # 0.1만큼 평행이동
    height_shift_range=0.1,     # 0.1만큼 수직이동
    rotation_range=5,           # 정해진 각도만큼 이미지를 회전
    zoom_range=1.2,             # 1.2배 확대(축소도 가능)
    shear_range=0.7,            # 좌표 하나를 고정시키고 다른 몇 개의 좌표를 이동시키는 변환
    fill_mode='nearest',        # 비어있는 데이터에 근처 가장 비슷한 값으로 변환(채워줌)                
)

augment_size = 500

randidx = np.random.randint(1027, size= augment_size) 
    
X_augmented = X[randidx].copy()               
y_augmented = y[randidx].copy()     

# print(X_augmented.shape, y_augmented.shape)    
X_augmented = train_datagen.flow(
    X_augmented,y_augmented,
    batch_size= augment_size,
    shuffle=False       
).next()[0]   
    
X_combined = np.concatenate((X, X_augmented))            
y_combined = np.concatenate((y, y_augmented))


X_train, X_test, y_train, y_test = train_test_split(X_combined , y_combined, test_size=0.3, shuffle=True, random_state=3, stratify=y_combined)


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
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
es = EarlyStopping(monitor='val_loss', mode='min', patience=50, verbose=2, restore_best_weights=True)
model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=2, callbacks=[es], validation_split=0.15)
end_time = time.time()

loss = model.evaluate(X_test, y_test)
print("loss : ", loss[0])
print("acc : ", loss[1])
print("걸린시간 : ", round(end_time - strat_time, 3), "초")



# loss :  0.32193633913993835
# acc :  0.8714597225189209
# 걸린시간 :  5.377 초




# loss :  0.24141526222229004
# acc :  0.9059602618217468
# 걸린시간 :  9.277 초





