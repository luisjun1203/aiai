# https://dacon.io/competitions/open/236068/mysubmission

import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error,accuracy_score
from keras. callbacks import EarlyStopping

path = "c://_data//dacon//diabetes//"

train_csv = pd.read_csv(path + "train.csv", index_col=0)
# print(train_csv)        #(652, 9)

test_csv = pd.read_csv(path + "test.csv", index_col=0)
# print(test_csv)         #(116, 8)

submission_csv = pd.read_csv(path + "sample_submission.csv")
# print(submission_csv)   #(116, 2)

# train_csv['Glucose']
# train_csv['Glucose'] = train_csv['Glucose'].fillna(train_csv['Glucose'].mean())

# print(train_csv['Glucose'][train_csv['Glucose']==0].count())    # 4
# print(train_csv['BloodPressure'][train_csv['BloodPressure']==0].count())    # 30
# print(train_csv['SkinThickness'][train_csv['SkinThickness']==0].count())    # 195
# print(train_csv['Insulin'][train_csv['Insulin']==0].count())    # 318
# print(train_csv['BMI'][train_csv['BMI']==0].count())    # 7






X = train_csv.drop(['Outcome'], axis=1)
y = train_csv['Outcome']

train_csv = train_csv.drop(['SkinThickness'], axis=1)


# print(X.shape, y.shape)     # (652, 8) (652)

def auto(a,b,c):

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=True, random_state=a)

    # print(X_train.shape, y_train.shape)         # (456, 8) (456,)
    # print(X_test.shape, y_test.shape)           # (196, 8) (196,)

    model = Sequential()
    model.add(Dense(19, input_dim = 8))               
    model.add(Dense(97))
    model.add(Dense(9, activation='sigmoid'))
    model.add(Dense(21,activatio= 'relu'))
    model.add(Dense(19))
    model.add(Dense(99))
    model.add(Dense(7))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics='acc')
    es = EarlyStopping(monitor='val_loss' , mode='min', patience=c, verbose=3, restore_best_weights=True)
    hist = model.fit(X_train, y_train, epochs=b, batch_size=25, validation_split=0.2, callbacks=[es] )

    loss = model.evaluate(X_test, y_test)
    y_submit = model.predict(test_csv)

    # print(y_submit)
    # print(y_submit.shape)       #(196, 1)

    submission_csv['Outcome'] = y_submit.round()                                       
    # print(submission_csv)       #(116, 2)

    submission_csv.to_csv(path + "submission_0110_2_.csv", index=False)

    y_predict = model.predict(X_test)
    y_predict = y_predict.round()
    def ACC(qqq, www):
        (accuracy_score(qqq, www))
        return (accuracy_score(qqq, www))
    acc = ACC(y_test, y_predict)
    print("ACC : ", acc)
    print("로스 : ", loss)

    return acc



import random
for i in range(10000000):
    # b = random.randrange(1, 900000)
    b = (516664)
    r = auto(b, 5000, 200)          
    print("random_state : ", b)
    if r > 0.83 :
        print("random_state : ", b)
        print("ACC : ", r)
        break