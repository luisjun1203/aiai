
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error,accuracy_score
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
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

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=3)      #58356
################    MinMaxScaler    ##############################
# mms = MinMaxScaler()
# mms.fit(X_train)
# X_train = mms.transform(X_train)
# X_test = mms.transform(X_test)

################    StandardScaler    ##############################

# sts = StandardScaler()
# sts.fit(X_train)
# X_train = sts.transform(X_train)
# X_test = sts.transform(X_test)

# print(X_train)
# print(X_test)

# ################    MaxAbsScaler    ##############################
mas = MaxAbsScaler()
mas.fit(X_train)
X_train = mas.transform(X_train)
X_test = mas.transform(X_test)
test_csv = mas.transform(test_csv)


# ################    RobustScaler    ##############################
# rbs = RobustScaler()
# rbs.fit(X_train)
# X_train = rbs.transform(X_train)
# X_test = rbs.transform(X_test)
# 2.모델구성

model = Sequential()
model.add(Dense(19,input_dim=9))
model.add(Dense(97, activation='relu'))
model.add(Dense(9))
model.add(Dense(21, activation='relu'))
model.add(Dense(19))
model.add(Dense(8, activation='relu'))
model.add(Dense(1))


from keras.optimizers import Adam
learning_rate = 0.01
epochs = 500
# 3.컴파일, 훈련
import datetime
date = datetime.datetime.now()
print(date)                     # 2024-01-17 10:52:59.857389
date = date.strftime("%m%d_%H%M")                   # _는 str      
print(date)                     # 0117_1058
print(type(date))               # <class 'str'>

path = "..\\_data\\_save\\MCP\\"
filename = '{epoch:05d}-{val_loss:.4f}-{loss:.4f}.hdf5'            # 04d : 4자리 정수표현, 4f : 소수4번째자리까지 표현, 예) 1000_0.3333.hdf5
filepath = "".join([path, 'k64_dacon_loan_',date,'_', filename])
mcp = ModelCheckpoint(monitor='val_loss', mode='min', verbose=1, save_best_only=True, filepath=filepath)    
rlr = ReduceLROnPlateau(monitor='val_loss', patience=10, mode='accuracy', verbose=1, factor=0.5)

model.compile(loss='mse', optimizer=Adam(learning_rate=learning_rate), metrics='accuracy')
es = EarlyStopping(monitor='val_loss', mode='min', patience=50, verbose=1, restore_best_weights=True)
hist = model.fit(X_train, y_train, epochs=epochs, batch_size=32, validation_split=0.15, verbose=2, callbacks=[es, mcp, rlr])

# 4.평가, 예측

loss = model.evaluate(X_test, y_test)
y_submit = model.predict(test_csv)
# print(y_submit)
# print(y_submit.shape)       # (715, 1)

# print("============================================")
######### submission.csv 만들기 (count컬럼에 값만 넣어줘) #######################
submission_csv['count'] = y_submit
# print(submission_csv)
# print(submission_csv.shape)

submission_csv.to_csv(path + "submission_0312_1.csv", index=False)
y_predict = model.predict(X_test) 
def RMSE(y_test, y_predict):
    np.sqrt(mean_squared_error(y_test, y_predict))
    return np.sqrt(mean_squared_error(y_test, y_predict))
rmse = RMSE(y_test, y_predict)
print("lr : {0}, epochs : {1} ,RMSE : {2}, 로스 : {3} ".format(learning_rate, epochs, rmse, loss))


# lr : 0.0001, epochs : 300 ,RMSE : 50.92841682414921, 로스 : [2593.70361328125, 0.0]
# lr : 0.001, epochs : 300 ,RMSE : 42.50620207319041, 로스 : [1806.7772216796875, 0.0]
# lr : 0.01, epochs : 300 ,RMSE : 42.5287314413487, 로스 : [1808.693115234375, 0.0]
# lr : 0.1, epochs : 300 ,RMSE : 47.818806885597, 로스 : [2286.638427734375, 0.0]
# lr : 1.0, epochs : 300 ,RMSE : 131.1556377982009, 로스 : [17201.798828125, 0.0]

# rlr 적용
# lr : 0.01, epochs : 500 ,RMSE : 39.96226189169073, 로스 : [1596.982421875, 0.0]