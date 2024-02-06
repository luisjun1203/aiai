import numpy as np
import pandas as pd
import time
from keras.models import Sequential, Model
from keras. layers import Dense, Conv1D, SimpleRNN, LSTM, Flatten, LSTM, Dropout, Input, concatenate
from keras. callbacks import EarlyStopping, ModelCheckpoint
from keras. utils import to_categorical
from sklearn.model_selection import train_test_split 
from sklearn.metrics import f1_score, mean_squared_error, accuracy_score, r2_score
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt

start_time = time.time()
path = "c:\\_data\\sihum\\"

ss_csv = pd.read_csv(path + "삼성 240205.csv", index_col=0, encoding='euc-kr')
am_csv = pd.read_csv(path + "아모레 240205.csv", index_col=0, encoding='euc-kr')


ss_csv = ss_csv.iloc[::-1].reset_index(drop=True)
am_csv = am_csv.iloc[::-1].reset_index(drop=True)


# print(ss_csv)
# print(am_csv)

ss_csv = ss_csv.drop(['전일비'], axis=1)
am_csv = am_csv.drop(['전일비'], axis=1)

ss_csv.rename(columns={'Unnamed: 6' : '전일비1'},inplace=True)
am_csv.rename(columns={'Unnamed: 6' : '전일비1'},inplace=True)

columns_to_convert = ['시가', '고가', '저가', '종가', '거래량', '금액(백만)', '개인', '기관', '외인(수량)', '외국계', '프로그램']

for column in columns_to_convert:
    ss_csv[column] = ss_csv[column].str.replace(',', '').astype(float)
    am_csv[column] = am_csv[column].str.replace(',', '').astype(float)
    
    
def convert_to_float(s):
    
    s = s.replace(',', '')
    return float(s)  

ss_csv['전일비1'] =ss_csv['전일비1'].apply(convert_to_float)
am_csv['전일비1'] =am_csv['전일비1'].apply(convert_to_float)
  

ss_csv['거래량'] = ss_csv['거래량'].fillna(0)
ss_csv['금액(백만)'] = ss_csv['금액(백만)'].fillna(method='ffill')
am_csv['거래량'] = am_csv['거래량'].fillna(0)
am_csv['금액(백만)'] = am_csv['금액(백만)'].fillna(0)


ss_csv.iloc[-1418:, ss_csv.columns.get_loc('거래량')] = ss_csv.iloc[-1418:, ss_csv.columns.get_loc('거래량')] / 50
# print(ss_csv['거래량'])

am_csv.iloc[-2154:, am_csv.columns.get_loc('거래량')] = am_csv.iloc[-2154:, am_csv.columns.get_loc('거래량')] / 10

ss_csv.iloc[-1418:, ss_csv.columns.get_loc('종가')] = ss_csv.iloc[-1418:, ss_csv.columns.get_loc('종가')] * 50
ss_csv.iloc[-1418:, ss_csv.columns.get_loc('시가')] = ss_csv.iloc[-1418:, ss_csv.columns.get_loc('시가')] * 50
ss_csv.iloc[-1418:, ss_csv.columns.get_loc('고가')] = ss_csv.iloc[-1418:, ss_csv.columns.get_loc('고가')] * 50
ss_csv.iloc[-1418:, ss_csv.columns.get_loc('저가')] = ss_csv.iloc[-1418:, ss_csv.columns.get_loc('저가')] * 50
# print(ss_csv['시가'])
# print(ss_csv['종가'])
# print(ss_csv['저가'])
# print(ss_csv['고가'])

am_csv.iloc[-2154:, am_csv.columns.get_loc('종가')] = am_csv.iloc[-2154:, am_csv.columns.get_loc('종가')] * 10
am_csv.iloc[-2154:, am_csv.columns.get_loc('시가')] = am_csv.iloc[-2154:, am_csv.columns.get_loc('시가')] * 10
am_csv.iloc[-2154:, am_csv.columns.get_loc('고가')] = am_csv.iloc[-2154:, am_csv.columns.get_loc('고가')] * 10
am_csv.iloc[-2154:, am_csv.columns.get_loc('저가')] = am_csv.iloc[-2154:, am_csv.columns.get_loc('저가')] * 10

mms = MinMaxScaler()
ss_csv = mms.fit_transform(ss_csv)
am_csv = mms.fit_transform(am_csv)

size = 180
def split_Xy(dataset, size, target_column):
    X, y = [], []
    for i in range(len(dataset) - size + 1 ):             
        X_subset = dataset[i : (i + size):, :]      # X에는 모든 행과, 첫 번째 컬럼부터 target_column 전까지의 컬럼을 포함
        y_subset = dataset[i + size - 1,  target_column]        # Y에는 모든 행과, target_column 컬럼만 포함 
        X.append(X_subset)
        y.append(y_subset)
    return np.array(X), np.array(y)

X , y = split_Xy(ss_csv, size, 0)

X1 = X[-1418:,]
X2 = X[-2836:-1418,]
y1 = y[-1418:,]

X , y = split_Xy(am_csv, size, 3)
X3 = X[-1418:,]
X4 = X[778:2196,]
y2 = y[-1418:,]

X_pre_ss = ss_csv[-180:].reshape(-1, 180, 15)  # 삼성전자 데이터
X_pre_am = am_csv[-180:].reshape(-1, 180, 15)

X1_train, X1_test, X2_train, X2_test, X3_train, X3_test, X4_train, X4_test, y1_train, y1_test, y2_train, y2_test = train_test_split(X1, X2, X3, X4, y1, y2,
                                                                                                               random_state=3, test_size=0.15 )

input_shape_rnn = (180,15)
rip = Input(shape=input_shape_rnn)
L1 = LSTM(19)(rip)
d2 = Dense(97, activation='swish')(L1)
d3 = Dense(9, activation='swish')(d2)
d4 = Dense(21, activation='swish')(d3)

rop = Dense(7, activation='swish')(d4)


input_shape_rnn2 = (180,15)
rip2 = Input(shape=input_shape_rnn2)
L11 = LSTM(19)(rip2)
d22 = Dense(99, activation='swish')(L11)
d33 = Dense(7, activation='swish')(d22)
d44 = Dense(13, activation='swish')(d33)

rop2 = Dense(7, activation='swish')(d44)

input_shape_rnn3 = (180,15)
rip3 = Input(shape=input_shape_rnn3)
L111 = LSTM(19)(rip3)
d222 = Dense(71, activation='swish')(L111)
d333 = Dense(11, activation='swish')(d222)
d444 = Dense(11, activation='swish')(d333)

rop3 = Dense(7, activation='swish')(d444)


input_shape_rnn4 = (180,15)
rip4 = Input(shape=input_shape_rnn4)
L1111 = LSTM(19)(rip4)
d2222 = Dense(66, activation='swish')(L1111)
d3333 = Dense(12, activation='swish')(d2222)
d4444 = Dense(11, activation='swish')(d3333)

rop4 = Dense(7, activation='swish')(d4444)

combined = concatenate([rop, rop2, rop3, rop4])



fl = Dense(6, activation='swish')(combined)
final_output = Dense(1)(fl)  
final_output2 = Dense(1)(fl)
model = Model(inputs=[rip, rip2, rip3, rip4], outputs=[final_output,final_output2])

model.summary()

import datetime
date = datetime.datetime.now()
print(date)                     
date = date.strftime("%m%d_%H%M")                        
print(date)                     

path2 = "c:\\_data\\sihum\\sihum2\\"
filename = '{epoch:05d}-{loss:.4f}.hdf5'            
filepath = "".join([path2, '02_05_sihum',date,'_', filename])

mcp = ModelCheckpoint(monitor='val_loss', mode='min', verbose=1, save_best_only=True, filepath=filepath)    
model.compile(loss='mae', optimizer='adam')

es = EarlyStopping(monitor='val_loss', mode='min', patience=300, verbose=20, restore_best_weights=True)
model.fit([X1_train, X2_train, X3_train,X4_train],[y1_train, y2_train], epochs=3000, validation_split=0.15, batch_size=100, verbose=2, callbacks=[es,mcp])

end = time.time()

results = model.evaluate([X1_test, X2_test,X3_test, X4_test], [y1_test, y2_test])
print("mae : ", results)
y_predict = model.predict([X1_test, X2_test,X3_test, X4_test])


y_predict_ss, y_predict_am = model.predict([X_pre_ss[0].reshape(1,180,15), X_pre_ss[0].reshape(1,180,15), X_pre_am[0].reshape(1,180,15), X_pre_am[0].reshape(1,180,15)])

y1_temp = np.zeros([1,15])
y1_temp[0][0] = y_predict_ss[0]

y2_temp = np.zeros([1,15])
y2_temp[0][0] = y_predict_am[0]

y_predict_ss = mms.inverse_transform(y1_temp)
y_predict_am = mms.inverse_transform(y2_temp)

print("2월 7일의 삼성전자  예측 시가 :", y_predict_ss[0][0]/50)
print("2월 7일의 아모레 예측 종가 : ", y_predict_am[0][0]/10)

r2_ss = r2_score(y1_test, y_predict[0])
r2_am = r2_score(y2_test, y_predict[1])
print("삼성 R2스코어 : ", r2_ss)
print("아모레 R2스코어 : ", r2_am)

# 내일 예측
# model.save("c:\\_data\\sihum\\02_06_save_model_01.h5") #상위폴더

# time step 180
# 삼성전자  예측가 : 73557.25578069687
# 아모레 예측가 :  122156.09486401081
# 삼성 R2스코어 :  0.9975698378074533
# 아모레 R2스코어 :  0.9907753148114041

# time step 180
# 삼성전자  예측가 : 72888.10650110245
# 아모레 예측가 :  131666.13756865263
# 삼성 R2스코어 :  0.9955882934180159
# 아모레 R2스코어 :  0.9901596827164

# time step 180
# 삼성전자  예측가 : 74551.39044523239
# 아모레 예측가 :  125831.28973096609
# 삼성 R2스코어 :  0.9973589372787095
# 아모레 R2스코어 :  0.9916116242190773


# 삼성전자  예측가 : 72864.15279507637
# 아모레 예측가 :  123638.3366882801
# 삼성 R2스코어 :  0.9975963448031284
# 아모레 R2스코어 :  0.990528740647695

