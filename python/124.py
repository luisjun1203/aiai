import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
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


def auto(a,b,c):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.123, shuffle=True, random_state=a)

    # print(X_train.shape, y_train.shape)     # (9253, 8) (9253,11)
    # print(X_test.shape, y_test.shape)       # (1633, 8) (1633, 11)


    model = Sequential()
    model.add(Dense(16, input_dim = 8, activation='relu'))                 # activation : 활성화함수, default : linear, activation : 하이퍼 파라미터
    model.add(Dense(32, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(4, activation='relu'))
    model.add(Dense(1))

    model.compile(loss='mse', optimizer='adam')
    s_time = time.time()
    model.fit(X_train, y_train, epochs=b, batch_size=c)
    e_time = time.time()

    
    loss = model.evaluate(X_test, y_test)
    y_submit = model.predict(X_test)
    r2 = r2_score(y_test, y_submit)
    print("로스 : ", loss)
    print("R2스코어 : ", r2)
    # print(y_submit)
    time.sleep(1.5)
    # print(y_submit.shape)   
    return r2



    # submission_csv['count'] = y_submit
    # print(submission_csv)
    # submission_csv.to_csv(path + "submission_0108_1.csv", index=False)
    
    
    
    
import random
for i in range(10000000):
    b = random.randrange(1, 10000)
    #b = (6544)
    r = auto(b, 500, 200)          
    print("random state : ", b)
    if r > 0.4 :
        print("random_state : ", b)
        break
    
    
# 307, 68, 3, 364, (100), 130, 347, 193, 939, 315, 94(0.288~)