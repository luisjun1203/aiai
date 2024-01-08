import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_log_error, mean_squared_error
import time



path = "c:\\_data\\kaggle\\bike\\"

train_csv = pd.read_csv(path + "train.csv", index_col=0)
# print(train_csv)        (10886,11)
test_csv = pd.read_csv(path + "test.csv", index_col=0)
# print(test_csv)       (6493, 8)
submission_csv = pd.read_csv(path + "samplesubmission.csv")
# print(submission_csv)

# print(train_csv.shape)              # (10886, 11)
# print(test_csv.shape)               # (6493, 8)
# print(submission_csv.shape)         # (6493, 2)

X = train_csv.drop(['casual', 'registered', 'count'], axis=1)
# print (X.shape)     # (10886, 8)
y = train_csv['count']      
print(y.shape)     # (10886, 8)

# df = pd.DataFrame(train_csv, columns = ['casual', 'registered', 'count'])
# list(df['count']>0)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, shuffle=True, random_state=713)

# print(X_train.shape, y_train.shape)     # (9253, 8) (9253,11)
# print(X_test.shape, y_test.shape)       # (1633, 8) (1633, 11)


model = Sequential()
model.add(Dense(16, input_dim = 8, activation='relu'))                 # activation : 활성화함수, default : linear, activation : 하이퍼 파라미터
model.add(Dense(32, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(4, activation='relu'))
model.add(Dense(1))


model.compile(loss='mse', optimizer='adam')
s_time = time.time()
model.fit(X_train, y_train, epochs=500, batch_size=200)
e_time = time.time()


loss = model.evaluate(X_test, y_test)
y_submit = model.predict(test_csv)
print(y_submit)
# print(y_submit.shape)   
# r2 = r2_score(y_test, y_submit)


submission_csv['count'] = y_submit                                        
print(submission_csv)
print("mse : ", loss)
# print("R2스코어 : ", r2)
submission_csv.to_csv(path + "submission_0108_10_2.csv", index=False)

print("음수갯수 : ",submission_csv[submission_csv['count']<0].count())      # 0보다 작은 조건의 모든 데이터셋을 세줘

y_predict = model.predict(X_test)                                           #rmse 구하기
# def RMSLE(y_test, y_predict):
#     np.sqrt(mean_squared_log_error(y_test, y_predict))
#     return np.sqrt(mean_squared_log_error(y_test, y_predict))
# rmsle = RMSLE(y_test, y_predict)

def RMSE(y_test, y_predict):
    np.sqrt(mean_squared_error(y_test, y_predict))
    return np.sqrt(mean_squared_error(y_test, y_predict))
rmse = RMSE(y_test, y_predict)
# print("256255음수갯수 : ",submission_csv[submission_csv['count']<0].count())      # 0보다 작은 조건의 모든 데이터셋을 세줘

# print("RMSLE : ", rmsle)
print("RMSE : ", rmse)



# 1 = 5287 epochs = 100
# 2 = 68
# 3 = 346
# 4 = 346 epochs = 1000
# 5 = 346 epochs = 1000 
# 6 = 94
# 7
# 8 = 5436














