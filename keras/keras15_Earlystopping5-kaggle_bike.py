import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_log_error, mean_squared_error
import time

path = "c:\\_data\\kaggle\\bike\\"

train_csv = pd.read_csv(path + "train.csv", index_col=0)

test_csv = pd.read_csv(path + "test.csv", index_col=0)

submission_csv = pd.read_csv(path + "samplesubmission.csv")

X = train_csv.drop(['casual', 'registered', 'count'], axis=1)

y = train_csv['count']      
print(y.shape)     # (10886, 8)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.385, shuffle=True, random_state=145534)

model = Sequential()
model.add(Dense(8, input_dim = 8, activation='relu'))                 # activation : 활성화함수, default : linear, activation : 하이퍼 파라미터
model.add(Dense(10, activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(60, activation='relu'))
model.add(Dense(120, activation='relu'))
model.add(Dense(40, activation='relu'))
model.add(Dense(6, activation='relu'))
model.add(Dense(1, activation='relu'))


model.compile(loss='mse', optimizer='adam')
s_time = time.time()
from keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss', mode='min', patience=400, verbose=3)
hist = model.fit(X_train, y_train, epochs=5000, batch_size=1200, validation_split=0.197, callbacks=[es])           # early stopping
e_time = time.time()


loss = model.evaluate(X_test, y_test)
y_submit = model.predict(test_csv)
print(y_submit)

submission_csv['count'] = y_submit                                        
print(submission_csv)
print("mse : ", loss)
# print("R2스코어 : ", r2)
submission_csv.to_csv(path + "submission_0109_val_3_.csv", index=False)

print("음수갯수 : ",submission_csv[submission_csv['count']<0].count())      # 0보다 작은 조건의 모든 데이터셋을 세줘

y_predict = model.predict(X_test)                                           #rmse 구하기
def RMSLE(y_test, y_predict):
    np.sqrt(mean_squared_log_error(y_test, y_predict))
    return np.sqrt(mean_squared_log_error(y_test, y_predict))
rmsle = RMSLE(y_test, y_predict)

def RMSE(y_test, y_predict):
    np.sqrt(mean_squared_error(y_test, y_predict))
    return np.sqrt(mean_squared_error(y_test, y_predict))
rmse = RMSE(y_test, y_predict)
# print("256255음수갯수 : ",submission_csv[submission_csv['count']<0].count())      # 0보다 작은 조건의 모든 데이터셋을 세줘

print("RMSLE : ", rmsle)
print("RMSE : ", rmse)