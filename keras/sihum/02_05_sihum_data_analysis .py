import numpy as np
import pandas as pd
import time
from keras.models import Sequential, Model
from keras. layers import Dense, Conv1D, SimpleRNN, LSTM, Flatten, GRU, Dropout, Input, concatenate
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
  

# print(ss_csv.info())
# print(am_csv.info())

#############################################삼성#################################################
#  #   Column  Non-Null Count  Dtype
# ---  ------  --------------  -----
#  0   시가      10296 non-null  float64
#  1   고가      10296 non-null  float64
#  2   저가      10296 non-null  float64
#  3   종가      10296 non-null  float64
#  4   전일비1    10296 non-null  float64
#  5   등락률     10296 non-null  float64
#  6   거래량     10292 non-null  float64
#  7   금액(백만)  10265 non-null  float64
#  8   신용비     10296 non-null  float64
#  9   개인      10296 non-null  float64
#  10  기관      10296 non-null  float64
#  11  외인(수량)  10296 non-null  float64
#  12  외국계     10296 non-null  float64
#  13  프로그램    10296 non-null  float64
#  14  외인비     10296 non-null  float64
# dtypes: float64(15)

############################################아모레######################################
#  #   Column  Non-Null Count  Dtype
# ---  ------  --------------  -----
#  0   시가      4350 non-null   float64
#  1   고가      4350 non-null   float64
#  2   저가      4350 non-null   float64
#  3   종가      4350 non-null   float64
#  4   전일비1    4350 non-null   float64
#  5   등락률     4350 non-null   float64
#  6   거래량     4340 non-null   float64
#  7   금액(백만)  4340 non-null   float64
#  8   신용비     4350 non-null   float64
#  9   개인      4350 non-null   float64
#  10  기관      4350 non-null   float64
#  11  외인(수량)  4350 non-null   float64
#  12  외국계     4350 non-null   float64
#  13  프로그램    4350 non-null   float64
#  14  외인비     4350 non-null   float64
# dtypes: float64(15)

# print(ss_csv.shape)     # (10296, 16)
# print(am_csv.shape)     # (4350, 16)


##############삼성 결측치#################################
# 시가         0
# 고가         0
# 저가         0
# 종가         0
# 전일비1       0
# 등락률        0
# 거래량        4
# 금액(백만)    31
# 신용비        0
# 개인         0
# 기관         0
# 외인(수량)     0
# 외국계        0
# 프로그램       0
# 외인비        0
# dtype: int64



################아모레 결측치####################
# 시가         0
# 고가         0
# 저가         0
# 종가         0
# 전일비1       0
# 등락률        0
# 거래량       10
# 금액(백만)    10
# 신용비        0
# 개인         0
# 기관         0
# 외인(수량)     0
# 외국계        0
# 프로그램       0
# 외인비        0
# dtype: int64






# print(ss_csv.describe())
# print(am_csv.describe())




# print(np.min(ss_csv['금액(백만)'])) # 1.0
# print(np.max(ss_csv['금액(백만)'])) # 8379238.0
# print(np.min(am_csv['금액(백만)'])) # 701.0
# print(np.max(am_csv['금액(백만)'])) # 667776.0

# print(np.unique(ss_csv['금액(백만)'], return_counts=True))
# print(np.unique(am_csv['금액(백만)'], return_counts=True))
# print(np.mean(ss_csv['금액(백만)']))    # 277029.4939113493
# print(np.mean(am_csv['금액(백만)']))    # 34261.43548387097

ss_csv['거래량'] = ss_csv['거래량'].fillna(0)
ss_csv['금액(백만)'] = ss_csv['금액(백만)'].fillna(method='ffill')
am_csv['거래량'] = am_csv['거래량'].fillna(0)
am_csv['금액(백만)'] = am_csv['금액(백만)'].fillna(0)


# print(ss_csv.isnull().sum())
# print(am_csv.isnull().sum())


# print(ss_csv.iloc[1417,0])  # 53000.0
# print(ss_csv.iloc[1418,0])  #2650000.0
# print(ss_csv['시가'].describe())
# print(am_csv.describe())
size = 180
def split_Xy(dataset, size, target_column):
    X, y = [], []
    for i in range(len(dataset) - size ):             
        X_subset = dataset.iloc[i : (i + size):, :]      # X에는 모든 행과, 첫 번째 컬럼부터 target_column 전까지의 컬럼을 포함
        y_subset = dataset.iloc[i + size ,  target_column]        # Y에는 모든 행과, target_column 컬럼만 포함 
        X.append(X_subset)
        y.append(y_subset)
    return np.array(X), np.array(y)

X , y = split_Xy(ss_csv, size, 0)

X1 = X[0:1418,]
X2 = X[1418:2836,]
y1 = y[0:1418]
y2 = y[1418: 2836,]
print(X1.shape)  # (1418, 180, 15)
print(X2.shape)  # (1418, 180, 15)
print(y.shape)  # (10116,)

X , y = split_Xy(am_csv, size, 0)
X3 = X[736:2154,]
X4 = X[2154:3572,]
# print(X3) # (1418, 180, 15)
print(X4) # (1418, 180, 15)




# print(X1.shape)

X1_train, X1_test, X2_train, X2_test, X3_train, X3_test, X4_train, X4_test, y1_train, y1_test, y2_train, y2_test = train_test_split(X1, X2, X3, X4, y1, y2,
                                                                                                               random_state=3, test_size=0.15 )

print(X1_train.shape)
print(X1_test.shape)

print(X2_train.shape)
print(X2_test.shape)

mms = MinMaxScaler()
X1_train = mms.fit_transform(X1_train.reshape(-1, 180*15)).reshape(-1, 180, 15)
X2_train = mms.fit_transform(X2_train.reshape(-1, 180*15)).reshape(-1, 180, 15)
X3_train = mms.fit_transform(X3_train.reshape(-1, 180*15)).reshape(-1, 180, 15)
X4_train = mms.fit_transform(X4_train.reshape(-1, 180*15)).reshape(-1, 180, 15)

X1_test = mms.fit_transform(X1_test.reshape(-1, 180*15)).reshape(-1, 180, 15)
X2_test = mms.fit_transform(X2_test.reshape(-1, 180*15)).reshape(-1, 180, 15)
X3_test = mms.fit_transform(X3_test.reshape(-1, 180*15)).reshape(-1, 180, 15)
X4_test = mms.fit_transform(X4_test.reshape(-1, 180*15)).reshape(-1, 180, 15)

input_shape_rnn = (180,15)
rip = Input(shape=input_shape_rnn)
L1 = GRU(19, activation='swish')(rip)
d2 = Dense(97, activation='swish')(L1)
d3 = Dense(9, activation='swish')(d2)
d4 = Dense(21, activation='swish')(d3)

rop = Dense(16, activation='swish')(d4)


input_shape_rnn2 = (180,15)
rip2 = Input(shape=input_shape_rnn2)
L11 = GRU(19, activation='swish')(rip2)
d22 = Dense(97, activation='swish')(L11)
d33 = Dense(9, activation='swish')(d22)
d44 = Dense(21, activation='swish')(d33)

rop2 = Dense(16, activation='swish')(d44)

input_shape_rnn3 = (180,15)
rip3 = Input(shape=input_shape_rnn3)
L111 = GRU(19, activation='swish')(rip3)
d222 = Dense(97, activation='swish')(L111)
d333 = Dense(9, activation='swish')(d222)
d444 = Dense(21, activation='swish')(d333)

rop3 = Dense(16, activation='swish')(d444)


input_shape_rnn4 = (180,15)
rip4 = Input(shape=input_shape_rnn4)
L1111 = GRU(19, activation='swish')(rip4)
d2222 = Dense(97, activation='swish')(L1111)
d3333 = Dense(9, activation='swish')(d2222)
d4444 = Dense(21, activation='swish')(d3333)

rop4 = Dense(16, activation='swish')(d4444)

combined = concatenate([rop, rop2, rop3, rop4])



fl = Dense(21, activation='swish')(combined)
final_output = Dense(1, activation='swish')(fl)  
final_output2 = Dense(1, activation='swish')(fl)
model = Model(inputs=[rip, rip2, rip3, rip4], outputs=[final_output,final_output2])

model.summary()

import datetime
date = datetime.datetime.now()
print(date)                     
date = date.strftime("%m%d_%H%M")                        
print(date)                     

path2 = "c:\\_data\\sihum\\"
filename = '{epoch:05d}-{loss:.4f}.hdf5'            
filepath = "".join([path2, '02_05_sihum',date,'_', filename])




mcp = ModelCheckpoint(monitor='val_loss', mode='min', verbose=1, save_best_only=True, filepath=filepath)    
model.compile(loss='mse', optimizer='adam')

es = EarlyStopping(monitor='val_loss', mode='min', patience=20, verbose=20, restore_best_weights=True)
model.fit([X1_train, X2_train, X3_train,X4_train],[y1_train, y2_train], epochs=100, validation_split=0.15, batch_size=1000, verbose=2, callbacks=[es,mcp])

end = time.time()

results = model.evaluate([X1_test, X2_test,X3_test,X4_test], [y1_test,y2_test])
print("mse : ", results)
y_predict = model.predict([X1_test, X2_test, X3_test,X4_test])
y_predict = y_predict.round(2)
print(y_predict)
r2 = r2_score([y1_test,y2_test], y_predict)
print("R2스코어 : ", r2)
