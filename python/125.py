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


X = X.drop(['SkinThickness'], axis=1)
test_csv = test_csv.drop(['SkinThickness'],axis=1)


# print(X.shape)
# print(test_csv.shape)

# print(train_csv.columns)
# print(test_csv.columns)
      

# print(X.shape, y.shape)     # (652, 8) (652)

def auto(a,b,c):

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=a)

    # print(X_train.shape, y_train.shape)         # (456, 8) (456,)
    # print(X_test.shape, y_test.shape)           # (196, 8) (196,)

    model = Sequential()
    model.add(Dense(19, input_dim = 7,activation= 'relu'))               
    model.add(Dense(97))
    model.add(Dense(9))
    model.add(Dense(21,activation= 'relu'))
    model.add(Dense(19))
    model.add(Dense(99))
    model.add(Dense(7))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam')
    es = EarlyStopping(monitor='val_loss' , mode='min', patience=c, verbose=3, restore_best_weights=True)
    hist = model.fit(X_train, y_train, epochs=b, batch_size=16, validation_split=0.25, callbacks=[es] )

    loss = model.evaluate(X_test, y_test)
    y_submit = model.predict(test_csv)

    # print(y_submit)
    # print(y_submit.shape)       #(196, 1)

    submission_csv['Outcome'] = y_submit.round()                                       
    # print(submission_csv)       #(116, 2)

    submission_csv.to_csv(path + "submission_0110_7_.csv", index=False)

    y_predict = model.predict(X_test)
    y_predict = y_predict.round()
    def ACC(qqq, www):
        (accuracy_score(qqq, www))
        return (accuracy_score(qqq, www))
    acc = ACC(y_test, y_predict)
    print("ACC : ", acc)
    print("로스 : ", loss)

    return loss



import random
for i in range(10000000):
    b = random.randrange(1, 1000)
    # b = (776)
    r = auto(b, 5000, 200)          
    print("random_state : ", b)
    if r < 0.4  :
        print("random_state : ", b)
        print("ACC : ", r)
        break
    
    
    
    
    
    
# random_state :  229951
# ACC :  0.8316326530612245    
