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




# xy_train = train_datagen.flow_from_directory(
#     path_train, 
#     target_size=(100,100),              # 사이즈 조절
#     batch_size=100,                      # 160이상을 쓰면 x 통데이터로 가져올 수 있다 , 몇 장씩 수치화 시킬건지 정해준다           
#     class_mode='binary',
#     shuffle=True)

path = "c:\\_data\\image\\cat_and_dog\\test\\"

np_path = "c:\\_data\\_save_npy\\"

# np.save(np_path + 'keras37_1_X_train.npy', arr=X)            # (160, 150, 150, 1) 이 데이터가   'keras39_1_X_train.npy 여기로 저장된다   
# np.save(np_path + 'keras37_1_y_train.npy', arr=y)  


X = np.load(np_path + 'keras37_3_X_train.npy')
y = np.load(np_path + 'keras37_3_y_train.npy')
test = np.load(np_path + 'keras37_3_test1.npy')


# print(X.shape)      # (19996, 150, 150, 3)
# print(y.shape)      # (19996,)
# print(test.shape)   # (5000, 100, 100, 3)

X_train, X_test, y_train, y_test = train_test_split(X , y, test_size=0.3, shuffle=True, random_state=3, stratify=y)
 



# 2. 모델 구성

model = Sequential()
model.add(Conv2D(19, (3,3), activation='swish', input_shape=(150, 150, 3)))
model.add(MaxPooling2D())
model.add(Conv2D(97, (4,4), activation='swish'))
model.add(MaxPooling2D())
model.add(Conv2D(9, (3,3), activation='swish'))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(21,activation='swish'))
model.add(Dense(1, activation='sigmoid'))

# model.summary()

# 3. 컴파일, 훈련

# strat_time = time.time()
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
es = EarlyStopping(monitor='val_loss', mode='min', patience=50, verbose=2, restore_best_weights=True)
model.fit(X_train, y_train, epochs=200, batch_size=10, verbose=2, callbacks=[es], validation_split=0.15)
# end_time = time.time()
# end_time = time.time()

# 4. 평가, 훈련
loss = model.evaluate(X_test, y_test)

y_predict = model.predict(test)
y_predict = np.around(y_predict.reshape(-1))
print(y_predict)
print(y_predict.shape)





forder_dir = path + "test1"
id_list = os.listdir(forder_dir)                    # listdir :  test1폴더에 있는 모든 파일 및 폴더의 목록을 가져온다
for i, id in enumerate(id_list):                    # enumerate : (id_list)를 감싸준다 인덱스와 원소로 이루어진 튜플을 만들어줌
    id_list[i] = int(id.split('.')[0])
    
# id.split('.'): 파일 이름에서 확장자를 제거 예를 들어, '123.jpg'의 경우 '123'만 추출
# int(id.split('.')[0]): 추출된 문자열을 정수로 변환
# id_list[i] = ...: 변환된 정수를 원래 목록에 다시 할당합니다. 이렇게 하면 파일 이름의 숫자 부분만 남게 된다


for id in id_list:
    print(id)


y_submit = pd.DataFrame({'id':id_list, 'Target':y_predict})     # 'id' : 이미지 파일의 숫자 부분을 나타냅니다. 'Target': 모델에서 예측한 결과인 y_predict를 나타냅니다.
print(y_submit)
print(y_submit.shape)
y_submit.to_csv(path + "submission_01_25_1.csv", index=False)

# def ACC(aaa, bbb):
#     (accuracy_score(aaa, bbb))
#     return (accuracy_score(aaa, bbb))
# acc = ACC(y_test, y_predict)



# print("ACC : ", acc)

# print("걸린시간 : ", round(end_time-start_time, 2),"초")

print("로스 : ", loss[0])
print("acc : ", loss[1])




















