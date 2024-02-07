
import numpy as np
import pandas as pd
from keras.models import Sequential, Model
from keras.layers import Dense, Conv2D, Flatten, Dropout, GlobalAveragePooling2D, MaxPooling2D
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error,accuracy_score
from keras. callbacks import EarlyStopping,ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, StandardScaler, RobustScaler
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

X = X.values.reshape(652, 2,2,2)
test_csv = test_csv.values.reshape(116,2,2,2)

# print(X.shape, y.shape)     # (652, 8) (652)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, shuffle=True, random_state=713)

################    MinMaxScaler    ##############################
# mms = MinMaxScaler()
# mms.fit(X_train)
# X_train = mms.transform(X_train)
# X_test = mms.transform(X_test)
# test_csv =mms.transform(test_csv)

################    StandardScaler    ##############################

# sts = StandardScaler()
# sts.fit(X_train)
# X_train = sts.transform(X_train)
# X_test = sts.transform(X_test)

# print(X_train)
# print(X_test)

# ################    MaxAbsScaler    ##############################
# mas = MaxAbsScaler()
# mas.fit(X_train)
# X_train = mas.transform(X_train)
# X_test = mas.transform(X_test)
# test_csv = mas.transform(test_csv)

# ################    RobustScaler    ##############################
# rbs = RobustScaler()
# rbs.fit(X_train)
# X_train = rbs.transform(X_train)
# X_test = rbs.transform(X_test)

# print(X_train.shape, y_train.shape)         # (456, 8) (456,)
# print(X_test.shape, y_test.shape)           # (196, 8) (196,)

# model = Sequential()
# model.add(Dense(19, input_dim = 8,activation='relu'))               
# model.add(Dense(97))
# model.add(Dense(9))
# model.add(Dense(21))
# model.add(Dense(3))
# model.add(Dense(41))
# model.add(Dropout(0.4))
# model.add(Dense(1, activation='sigmoid'))

# i1 = Input(shape = (8,))
# d1 = Dense(19)(i1)      
# d2 = Dense(97)(d1)
# d3 = Dense(9)(d2)
# d4 = Dense(21)(d3)
# d5 = Dense(3)(d4)
# d6 = Dense(41)(d5)
# drop1 = Dropout(0.4)(d6)
# o1 = Dense(1,activation='sigmoid')(drop1)
# model = Model(inputs = i1, outputs = o1)


model = Sequential()                    
model.add(Conv2D(19, kernel_size=(3, 3), input_shape=(2, 2, 2), activation='relu', strides=1, padding='same'))   
model.add(Conv2D(97, (2, 2), activation='relu',strides=1, padding='same'))                         
model.add(Conv2D(500, (3, 3), activation='relu', strides=2, padding='same'))              
model.add(GlobalAveragePooling2D())  
model.add(Dense(124, activation='relu'))
model.add(Dropout(0.3)) 
model.add(Dense(48, activation='relu'))
model.add(Dense(1,activation='sigmoid'))



import datetime
date = datetime.datetime.now()
# print(date)                     # 2024-01-17 10:52:59.857389
date = date.strftime("%m%d_%H%M")                   # _는 str      
# print(date)                     # 0117_1058
# print(type(date))               # <class 'str'>

path = "..\\_data\\_save\\MCP\\"
filename = '{epoch:05d}-{acc:.4f}-{loss:.4f}.hdf5'            # 04d : 4자리 정수표현, 4f : 소수4번째자리까지 표현, 예) 1000_0.3333.hdf5
filepath = "".join([path, 'k28_07_dacon_diabetes_',date,'_', filename])

# mcp = ModelCheckpoint(monitor='val_loss', mode='min', verbose=1, save_best_only=True, filepath=filepath)    
model.compile(loss='binary_crossentropy', optimizer='adam', metrics='acc')
es = EarlyStopping(monitor='val_loss' , mode='min', patience=100, verbose=3, restore_best_weights=True)
hist = model.fit(X_train, y_train, epochs=500, batch_size=16, validation_split=0.15, callbacks=[es] )

loss = model.evaluate(X_test, y_test)
y_submit = model.predict(test_csv)

# print(y_submit)
# print(y_submit.shape)       #(196, 1)

submission_csv['Outcome'] = y_submit.round()                                       
# print(submission_csv)       #(116, 2)

submission_csv.to_csv(path + "submission_0116_1_.csv", index=False)

y_predict = model.predict(X_test)
y_predict = y_predict.round()
def ACC(aaa, bbb):
    (accuracy_score(aaa, bbb))
    return (accuracy_score(aaa, bbb))
acc = ACC(y_test, y_predict)
print("ACC : ", acc)
print("로스 : ", loss)



# MinMaxScaler
# ACC :  0.8163265306122449
# 로스 :  [0.452159583568573, 0.8163265585899353]

# # MaxAbsScaler
# ACC :  0.8214285714285714
# 로스 :  [0.46231532096862793, 0.8214285969734192]
# # StandardScaler
# ACC :  0.8061224489795918
# 로스 :  [0.45180371403694153, 0.8061224222183228]

# # RobustScaler
# ACC :  0.8061224489795918
# 로스 :  [0.45696309208869934, 0.8061224222183228]

# ACC :  0.7791411042944786
# 로스 :  [0.49091923236846924, 0.7791411280632019]