
import numpy as np
import pandas as pd
from keras.models import Sequential, Model
from keras.layers import Dense, Conv2D, Flatten, Dropout, GlobalAveragePooling2D, MaxPooling2D
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error,accuracy_score
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, StandardScaler, RobustScaler
# 1.데이터

path = "c:\\_data\\dacon\\ddarung\\"

train_csv = pd.read_csv(path + "train.csv", index_col=0)        
                                             
test_csv = pd.read_csv(path + "test.csv", index_col=0)

submission_csv = pd.read_csv(path + "submission.csv")



train_csv['hour_bef_precipitation'] = train_csv['hour_bef_precipitation'].fillna(0)
train_csv['hour_bef_pm10'] = train_csv['hour_bef_pm10'].fillna(0)
train_csv['hour_bef_pm2.5'] = train_csv['hour_bef_pm2.5'].fillna(0)
train_csv['hour_bef_windspeed'] = train_csv['hour_bef_windspeed'].fillna(0)
train_csv['hour_bef_temperature'] = train_csv['hour_bef_temperature'].fillna(train_csv['hour_bef_temperature'].mean())
train_csv['hour_bef_humidity'] = train_csv['hour_bef_humidity'].fillna(train_csv['hour_bef_humidity'].mean())
train_csv['hour_bef_visibility'] = train_csv['hour_bef_visibility'].fillna(train_csv['hour_bef_visibility'].mean())
train_csv['hour_bef_ozone'] = train_csv['hour_bef_ozone'].fillna(train_csv['hour_bef_ozone'].mean())


test_csv = test_csv.fillna(test_csv.mean())
print(test_csv.info())      # 717 non-null

X = train_csv.drop(['count'], axis=1)

y = train_csv['count']

# print(X.shape)  # (1459, 9)
# print(y.shape)  # (1459,)
X = X.values.reshape(1459, 3, 3, 1)
# y = y.values.reshape()
# print(test_csv.shape)
test_csv = test_csv.values.reshape(715, 3, 3, 1)
# train_csv = train_csv.values.reshape(1459, 10, 1, 1)

print(train_csv.info())



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=3)      #58356





model = Sequential()                    
model.add(Conv2D(19, kernel_size=(3, 3), input_shape=(3, 3, 1), activation='relu', strides=1, padding='same'))   
model.add(Conv2D(97, (2, 2), activation='relu',strides=1, padding='same'))                         
model.add(Conv2D(500, (3, 3), activation='relu', strides=2, padding='same'))              
model.add(GlobalAveragePooling2D())  
model.add(Dense(124, activation='relu'))
model.add(Dropout(0.3)) 
model.add(Dense(48, activation='relu'))
model.add(Dense(1))



# 3.컴파일, 훈련
import datetime
date = datetime.datetime.now()
print(date)                     # 2024-01-17 10:52:59.857389
date = date.strftime("%m%d_%H%M")                   # _는 str      
print(date)                     # 0117_1058
print(type(date))               # <class 'str'>

path = "..\\_data\\_save\\MCP\\"
filename = '{epoch:05d}-{val_loss:.4f}-{loss:.4f}.hdf5'            # 04d : 4자리 정수표현, 4f : 소수4번째자리까지 표현, 예) 1000_0.3333.hdf5
filepath = "".join([path, 'k28_04_dacon_ddarung_',date,'_', filename])
# mcp = ModelCheckpoint(monitor='val_loss', mode='min', verbose=1, save_best_only=True, filepath=filepath)    
model.compile(loss='mse', optimizer='adam', metrics='accuracy')
es = EarlyStopping(monitor='val_loss', mode='min', patience=200, verbose=20, restore_best_weights=True)
hist = model.fit(X_train, y_train, epochs=1000, batch_size=32, validation_split=0.15, verbose=2, callbacks=[es])

# 4.평가, 예측

loss = model.evaluate(X_test, y_test)
y_submit = model.predict(test_csv)

submission_csv['count'] = y_submit

submission_csv.to_csv(path + "submission_0116_1.csv", index=False)
y_predict = model.predict(X_test) 
def RMSE(y_test, y_predict):
    np.sqrt(mean_squared_error(y_test, y_predict))
    return np.sqrt(mean_squared_error(y_test, y_predict))
rmse = RMSE(y_test, y_predict)
print("RMSE : ", rmse)
print("로스 : ", loss)


# RMSE :  43.594638008786504
# 로스 :  [1900.4923095703125, 0.0]


# RMSE :  49.395894309949746
# 로스 :  [2439.954345703125, 0.0]

