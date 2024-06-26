import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, accuracy_score, mean_squared_log_error
from keras.callbacks import EarlyStopping,ModelCheckpoint, ReduceLROnPlateau
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, StandardScaler, RobustScaler

path = "c:\\_data\\kaggle\\bike\\"

train_csv = pd.read_csv(path + "train.csv" , index_col=0)
test_csv = pd.read_csv(path + "test.csv", index_col=0)
submission_csv = pd.read_csv(path + "samplesubmission.csv")

X = train_csv.drop(['casual', 'registered', 'count'], axis=1) 
y = train_csv['count']

# print(X.shape)      #(10886, 8)
# print(y.shape)      #(10886)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=True, random_state=713)

#############    MinMaxScaler    ##############################
mms = MinMaxScaler()
mms.fit(X_train)
X_train = mms.transform(X_train)
X_test = mms.transform(X_test)
test_csv = mms.transform(test_csv)

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


# ################    RobustScaler    ##############################
# rbs = RobustScaler()
# rbs.fit(X_train)
# X_train = rbs.transform(X_train)
# X_test = rbs.transform(X_test)

model = Sequential()
model.add(Dense(19, input_shape= (8, ),activation='relu'))
model.add(Dense(97))
model.add(Dense(9,activation='relu'))
model.add(Dense(21))
model.add(Dense(16,activation='relu'))
model.add(Dense(21))
model.add(Dense(1, activation='relu'))


from keras.optimizers import Adam
learning_rate = 0.0001
epochs = 300
rlr = ReduceLROnPlateau(monitor='val_loss', patience=10, mode='accuracy', verbose=1, factor=0.5)
   
import datetime
date = datetime.datetime.now()
print(date)                     # 2024-01-17 10:52:59.857389
date = date.strftime("%m%d_%H%M")                   # _는 str      
print(date)                     # 0117_1058
print(type(date))               # <class 'str'>

path = "..\\_data\\_save\\MCP\\"
filename = '{epoch:05d}-{val_loss:.4f}-{loss:.4f}.hdf5'            # 04d : 4자리 정수표현, 4f : 소수4번째자리까지 표현, 예) 1000_0.3333.hdf5
filepath = "".join([path, 'k63_kaggle_bike_',date,'_', filename])
mcp = ModelCheckpoint(monitor='val_loss', mode='min', verbose=1, save_best_only=True, filepath=filepath)    
   
model.compile(loss='mse', optimizer = Adam(learning_rate=learning_rate), metrics='accuracy')
es = EarlyStopping(monitor='val_loss', mode='min', patience=50, verbose=1, restore_best_weights=True)
model.fit(X_train, y_train, epochs= epochs, batch_size=700, validation_split=0.15,callbacks=[es, mcp, rlr])


loss = model.evaluate(X_test, y_test)
y_submit = model.predict(test_csv)
y_predict = model.predict(X_test)
submission_csv['count'] = y_submit
# print("mse : ",loss )
submission_csv.to_csv(path + "submission_0312_1.csv", index=False)

# print("음수 : ", submission_csv[submission_csv['count']<0].count())

r2 = r2_score(y_test, y_predict)
def RMSLE(y_test, y_predict):
    np.sqrt(mean_squared_log_error(y_test, y_predict))
    return np.sqrt(mean_squared_log_error(y_test, y_predict))
rmsle = RMSLE(y_test, y_predict) 

# print("RMSLE : ", rmsle)
print("lr : {0}, epochs : {1}, RMSLE : {2}, 로스 : {3} ".format(learning_rate, epochs, rmsle, loss ))


# lr : 1.0, epochs : 300, RMSLE : 4.792837657395268, 로스 : [69748.8828125, 0.0]
# lr : 0.1, epochs : 300, RMSLE : 1.5816361405135992, 로스 : [33388.2578125, 0.0101041030138731]
# lr : 0.01, epochs : 300, RMSLE : 1.3358960483983204, 로스 : [22479.046875, 0.0101041030138731] 
# lr : 0.001, epochs : 300, RMSLE : 1.3526106816916024, 로스 : [22798.1953125, 0.0101041030138731]
# lr : 0.0001, epochs : 300, RMSLE : 1.3701661677225379, 로스 : [24140.763671875, 0.0101041030138731]

# rlr 적용
# lr : 0.0001, epochs : 300, RMSLE : 1.3650784151801927, 로스 : [24164.01953125, 0.0101041030138731]

# # MinMaxScaler
# RMSLE :  4.800237655708398

# # MaxAbsScaler
# RMSLE :  1.305517434934688
# # StandardScaler
# RMSLE :  1.3162207010884128

# # RobustScaler
# RMSLE :  1.2885431257772446


