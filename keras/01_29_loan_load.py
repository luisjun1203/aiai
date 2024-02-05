import numpy as np
import pandas as pd
from keras.models import Sequential, Model, load_model
from keras.layers import Dense, Dropout,Input, Conv2D, MaxPooling2D, Flatten, GlobalAveragePooling2D,concatenate, Reshape
from keras. callbacks import EarlyStopping, ModelCheckpoint
from keras. utils import to_categorical
from sklearn.model_selection import train_test_split 
from sklearn.metrics import f1_score
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler
import time
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import os

path = "c:\\_data\\dacon\\loan\\"

train_csv = pd.read_csv(path + "train.csv", index_col=0)
test_csv = pd.read_csv(path + "test.csv", index_col=0)
submission_csv = pd.read_csv(path + "sample_submission.csv")

test_csv.loc[test_csv['대출기간']==' 36 months', '대출기간'] =36
train_csv.loc[train_csv['대출기간']==' 36 months', '대출기간'] =36

test_csv.loc[test_csv['대출기간']==' 60 months', '대출기간'] =60
train_csv.loc[train_csv['대출기간']==' 60 months', '대출기간'] =60
 
test_csv.loc[test_csv['근로기간']=='3', '근로기간'] ='3 years'
train_csv.loc[train_csv['근로기간']=='3', '근로기간'] ='3 years'
test_csv.loc[test_csv['근로기간']=='1 year','근로기간']='1 years'
train_csv.loc[train_csv['근로기간']=='1 year','근로기간']='1 years'
test_csv.loc[test_csv['근로기간']=='<1 year','근로기간']='< 1 year'
train_csv.loc[train_csv['근로기간']=='<1 year','근로기간']='< 1 year'
test_csv.loc[test_csv['근로기간']=='10+years','근로기간']='10+ years'
train_csv.loc[train_csv['근로기간']=='10+years','근로기간']='10+ years'
train_csv.loc[train_csv['근로기간']=='Unknown', '근로기간']='10+ years'
test_csv.loc[test_csv['근로기간']=='Unknown', '근로기간']='10+ years'
train_csv.value_counts('근로기간')
 
 
train_csv.loc[train_csv['주택소유상태']=='ANY', '주택소유상태'] = 'OWN'

test_csv.loc[test_csv['대출목적']=='결혼', '대출목적'] = '기타'


lae = LabelEncoder()

lae.fit(train_csv['주택소유상태'])
train_csv['주택소유상태'] = lae.transform(train_csv['주택소유상태'])
test_csv['주택소유상태'] = lae.transform(test_csv['주택소유상태'])


lae.fit(train_csv['대출목적'])
train_csv['대출목적'] = lae.transform(train_csv['대출목적'])
test_csv['대출목적'] = lae.transform(test_csv['대출목적'])


lae.fit(train_csv['근로기간'])
train_csv['근로기간'] = lae.transform(train_csv['근로기간'])
test_csv['근로기간'] = lae.transform(test_csv['근로기간'])



X = train_csv.drop(['대출등급'], axis=1)
y = train_csv['대출등급']

y = y.values.reshape(-1, 1)
ohe = OneHotEncoder(sparse=False)
ohe.fit(y)
y1 = ohe.transform(y)


X_train, X_test, y_train, y_test = train_test_split(X, y1, test_size=0.15, shuffle=True, random_state=3721905478, stratify=y1)
# start = time.time()

# X_train = np.asarray(X_train).astype(np.float32)
# X_test = np.asarray(X_test).astype(np.float32)
# test_csv = np.asarray(test_csv).astype(np.float32)

# rbs = RobustScaler(quantile_range=(15,85))
# rbs.fit(X_train)
# X_train = rbs.transform(X_train)
# X_test = rbs.transform(X_test)
# test_csv = rbs.transform(test_csv)






# mms1 = ['대출기간',
#         '대출금액',
#         '연간소득',
#         '부채_대비_소득_비율',
#         '총계좌수',
#         '총상환원금',
#         '총상환이자'
#         ]

# mms = MinMaxScaler()
# mms.fit(X_train[mms1])
# X_train[mms1] = mms.transform(X_train[mms1])
# X_test[mms1] = mms.transform(X_test[mms1])
# test_csv[mms1] = mms.transform(test_csv[mms1])

mms = MinMaxScaler()
mms.fit(X_train)
X_train = mms.transform(X_train)
X_test = mms.transform(X_test)
test_csv = mms. transform(test_csv)


# # print(np.unique(X_train[mms1],return_counts=True))
# rbs1 = [
#     '연체계좌수', 
#         '총연체금액', 
#         '최근_2년간_연체_횟수'
#         ]

# rbs = RobustScaler()
# rbs.fit(X_train[rbs1])
# X_train[rbs1] = rbs.transform(X_train[rbs1])
# X_test[rbs1] = rbs.transform(X_test[rbs1])
# test_csv[rbs1] = rbs.transform(test_csv[rbs1])

# print(np.unique(X_train[rbs1], return_counts = True)


X_train_dnn = X_train.reshape(-1, 13)  
X_test_dnn = X_test.reshape(-1, 13) 
test_csv_dnn = test_csv.reshape(-1, 13)

X_train_dnn2 = X_train.reshape(-1, 13) 
X_test_dnn2 = X_test.reshape(-1, 13) 
test_csv_dnn2 = test_csv.reshape(-1, 13)

input_shape_dnn = (13,)
dip = Input(shape=input_shape_dnn)
d1 = Dense(19, activation='swish')(dip)
d2 = Dense(97, activation='swish')(d1)
d3 = Dense(9, activation='swish')(d2)
d4 = Dense(21,activation='swish')(d3)
dop = Dense(16, activation='swish')(d4)


# input_shape_dnn2 = (13,)
# dip1 = Input(shape = input_shape_dnn2) 
# d11 = Dense(50, activation='swish')(dip1)
# d22 = Dense(10, activation='swish')(d11)
# d33 = Dense(80, activation='swish')(d22)
# # drop1 = Dropout(0.4)(d33)
# d44 = Dense(10, activation='swish')(d33)
# d55 = Dense(70, activation='swish')(d44)
# # drop2 = Dropout(0.4)(d55)
# d66 = Dense(10, activation='swish')(d55)
# d77 = Dense(60, activation='swish')(d66)
# # drop3 = Dropout(0.4)(d77)
# d88 = Dense(10, activation='swish')(d77)
# d99 = Dense(50, activation='swish')(d88)
# # drop4 = Dropout(0.4)(d99)
# d10 = Dense(10, activation='swish')(d99)
# d12 = Dense(40, activation='swish')(d10)
# # drop5 = Dropout(0.4)(d12)
# d13 = Dense(10, activation='swish')(d12)
# dop1 = Dense(50, activation='swish')(d13)

# combined = concatenate([dop, dop1])

# fl = Dense(21, activation='swish')(combined)
# final_output = Dense(7, activation='softmax')(fl)  

# model = Model(inputs=[dip, dip1], outputs=final_output)

# model.summary()

input_shape_dnn = (13,)
dip2 = Input(shape=input_shape_dnn)
d11 = Dense(19, activation='swish')(dip)
d22 = Dense(99, activation='swish')(d11)
d33 = Dense(7, activation='swish')(d22)
d44 = Dense(13, activation='swish')(d33)

dop2 = Dense(16, activation='swish')(d44)

combined = concatenate([dop, dop2])



fl = Dense(21, activation='swish')(combined)
final_output = Dense(7, activation='softmax')(fl)  

model = Model(inputs=[dip, dip2], outputs=final_output)




import datetime
date = datetime.datetime.now()
print(date)                     
date = date.strftime("%m%d_%H%M")                        
print(date)                     

path = "..\\_data\\_save\\MCP\\"
filename = '{epoch:05d}-{acc:.4f}-{loss:.4f}.hdf5'            
filepath = "".join([path, 'k30_3_dacon_loan_',date,'_', filename])




mcp = ModelCheckpoint(monitor='val_loss', mode='min', verbose=1, save_best_only=True, filepath=filepath)    
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics='acc')

es = EarlyStopping(monitor='val_loss', mode='min', patience=1500, verbose=20, restore_best_weights=True)
model.fit([X_train_dnn, X_train_dnn2], y_train, epochs=20000, batch_size=502, validation_split=0.15, verbose=2, callbacks=[es])




# model= load_model("c:\\_data\\_save\\dacon_loan_2\\dacon_loan_1_auto_rs_3721905478_bs_502_f1_0.9334.h5")


results = model.evaluate([X_test_dnn, X_test_dnn2], y_test)
print("ACC : ", results[1])

y_predict = model.predict([X_test_dnn, X_test_dnn2]) 
y_test = ohe.inverse_transform(y_test)
y_predict = ohe.inverse_transform(y_predict)


y_submit = model.predict([test_csv_dnn, test_csv_dnn2])  
y_submit = ohe.inverse_transform(y_submit)

y_submit = pd.DataFrame(y_submit)
submission_csv['대출등급'] = y_submit
# print(y_submit)

fs = f1_score(y_test, y_predict, average='weighted')
print("f1_score : ", fs)
# print("걸린시간 : ",round(end - start, 3), "초")
submission_csv.to_csv(path + "submission_02_05_1_.csv", index=False)
print(y_submit)


















