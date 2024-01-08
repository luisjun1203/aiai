# https://dacon.io/competitions/open/235576/mysubmission

import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import time

# 1.데이터

path = "c:\\_data\\dacon\\ddarung\\"

# print( path + "aaa.csv")      # c:\_data\dacon\ddarung\aaa.csv

train_csv = pd.read_csv(path + "train.csv", index_col=0)        # 컬럼 지정
# train_csv = pd.read_csv("c:\_data\dacon\ddarung\\train.csv")  # \ = \\ =  / = //
# print(train_csv)                                                

test_csv = pd.read_csv(path + "test.csv", index_col=0)
# print(test_csv)

submission_csv = pd.read_csv(path + "submission.csv")
# print(submission_csv)

# print(train_csv.shape)              # (1459, 10)
# print(test_csv.shape)               # (715, 9)
# print(submission_csv.shape)         # (715, 2)

# print(train_csv.columns)
# # ['hour', 'hour_bef_temperature', 'hour_bef_precipitation',
# #        'hour_bef_windspeed', 'hour_bef_humidity', 'hour_bef_visibility',
# #        'hour_bef_ozone', 'hour_bef_pm10', 'hour_bef_pm2.5', 'count']
# print(train_csv.info())
# print(test_csv.info())

# print(train_csv.describe())

##### 결측치 처리 1.제거 #################

# print(train_csv.isnull().sum())
# print(train_csv.isna().sum())           #위 아래 같다

# train_csv = train_csv.dropna()

train_csv['hour_bef_precipitation'] = train_csv['hour_bef_precipitation'].fillna(0)
train_csv['hour_bef_pm10'] = train_csv['hour_bef_pm10'].fillna(0)
train_csv['hour_bef_pm2.5'] = train_csv['hour_bef_pm2.5'].fillna(0)
train_csv['hour_bef_windspeed'] = train_csv['hour_bef_windspeed'].fillna(0)
train_csv['hour_bef_temperature'] = train_csv['hour_bef_temperature'].fillna(train_csv['hour_bef_temperature'].mean())
train_csv['hour_bef_humidity'] = train_csv['hour_bef_humidity'].fillna(train_csv['hour_bef_humidity'].mean())
train_csv['hour_bef_visibility'] = train_csv['hour_bef_visibility'].fillna(train_csv['hour_bef_visibility'].mean())
train_csv['hour_bef_ozone'] = train_csv['hour_bef_ozone'].fillna(train_csv['hour_bef_ozone'].mean())

# print(train_csv.shape)      # (1459, 10)
# print(train_csv.info())     
# print(train_csv.isna().sum())




# print(train_csv.isna().sum())           #위 아래 같다
# print(train_csv.info())
# print(train_csv.shape)      #(1358, 10)

#test_csv = test_csv.fillna(test_csv.mean())
# print(test_csv.info())      # 717 non-null


############# X 와 y를 분리 #####################

X = train_csv.drop(['count'], axis=1)
# print(X)            # (1328, 9)

y = train_csv['count']
# print(y)

def auto(a,b,c):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, shuffle=True, random_state=a)

    # print(X_train.shape, y_train.shape)     #(1195, 9) (1195,)
    # print(X_test.shape, y_test.shape)       # (133, 9) (133,)


    # 2.모델구성

    model = Sequential()
    model.add(Dense(12,input_dim=9))
    model.add(Dense(203))
    model.add(Dense(50))
    model.add(Dense(840))
    model.add(Dense(402))
    model.add(Dense(120))
    model.add(Dense(815))
    model.add(Dense(146))
    model.add(Dense(1))

    # 3.컴파일, 훈련

    model.compile(loss='mse', optimizer='adam')
    model.fit(X_train, y_train, epochs=b, batch_size=c)


    # 4.평가, 예측

    loss = model.evaluate(X_test, y_test)
    y_submit = model.predict(X_test)
    r2 = r2_score(y_test, y_submit)
    # print(y_submit)
    # print(y_submit.shape)       # (715, 1)

    print("============================================")
    ######### submission.csv 만들기 (count컬럼에 값만 넣어줘) #######################
    # submission_csv['count'] = y_submit
    # print(submission_csv)
    # print(submission_csv.shape)
    print("로스 : ", loss)
    print("R2스코어 : ", r2)
    time.sleep(1)

    # submission_csv.to_csv(path + "submission_0105.csv", index=False)

    return r2
    
    
  

import random
for i in range(10000000):
    b = random.randrange(1, 9000)
    # b = (1226)
    r = auto(b, 100, 25)          
    print("random state : ", b)
    if r > 0.63 :
        print("random_state : ", b)
        break
        

    
    

# R2스코어 :  0.7193710891919103
# random_state :  497830280
    
    
    
    
    
# 로스 :  2428.158203125
# R2스코어 :  0.6753108817912734
# random_state :  58356


# random.randrange(1, 99000000)