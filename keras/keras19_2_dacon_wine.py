# https://dacon.io/competitions/open/235610/mysubmission


import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from keras.utils import to_categorical


path = "c:\\_data\\dacon\\wine\\"

train_csv = pd.read_csv(path + "train.csv", index_col=0)        
                                             
test_csv = pd.read_csv(path + "test.csv", index_col=0)

submission_csv = pd.read_csv(path + "sample_submission.csv")

# print(train_csv)
# print(train_csv.shape)      #(5497, 13)
# print(test_csv)
# print(test_csv.shape)       #(1000, 12)


# print(X.shape)      #(5497, 12)
# print(y)
# print(y.shape)      #(5497, )




     
lae = LabelEncoder()
lae.fit(train_csv['type'])
train_csv['type'] = lae.transform(train_csv['type'])
test_csv['type'] = lae.transform(test_csv['type'])

X = train_csv.drop(['quality'], axis=1)
y = train_csv['quality']

y = pd.get_dummies(y)

print(y)
print(y.shape)      #(5497, 7)      # 3,4,5,6,7,8,9

# print(X)

# print(X.shape)          # (5497, 12)
# print(y.shape)          #(5497, 7)
# print(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.45, shuffle=True, random_state=781, stratify=y)       #9266, 781

model = Sequential()
model.add(Dense(19, input_dim=12,activation='relu'))
model.add(Dense(97))             
model.add(Dense(9, activation='relu'))      
model.add(Dense(21, activation='relu'))      
model.add(Dense(4,activation='relu'))      
model.add(Dense(19, activation='relu'))      
model.add(Dense(28))      
model.add(Dense(7, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics='acc')
es = EarlyStopping(monitor='val_loss', mode='min', patience=200, verbose=3, restore_best_weights=True)
model.fit(X_train, y_train, epochs=3000, batch_size=8, validation_split=0.25, callbacks=[es], verbose=2)





results = model.evaluate(X_test, y_test)
print("ACC : ", results[1])

  

y_submit = model.predict(test_csv)  
y_predict = model.predict(X_test) 

y_test = np.argmax(y_test, axis=1)
y_predict = np.argmax(y_predict, axis=1)
y_submit = np.argmax(y_submit, axis=1)+3

submission_csv['quality'] = y_submit

# print(y_submit)
# print(y_submit.shape) 


submission_csv.to_csv(path + "submission_0112_5_.csv", index=False)

acc = accuracy_score(y_predict, y_test)
print("accuracy_score : ", acc)
print("로스 : ", results[0])
















