import numpy as np
import pandas as pd
import time
from keras.models import Sequential, Model, load_model
from keras. layers import Dense, Conv1D, SimpleRNN, LSTM, Flatten, GRU, Dropout, Input, concatenate
from keras. callbacks import EarlyStopping, ModelCheckpoint
from keras. utils import to_categorical
from sklearn.model_selection import train_test_split 
from sklearn.metrics import f1_score, mean_squared_error, accuracy_score, r2_score
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import keras
import tensorflow as tf
import random as rn

tf.random.set_seed(3)       
np.random.seed(3)
rn.seed(3)



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

am_csv.iloc[-2154:, am_csv.columns.get_loc('거래량')] = am_csv.iloc[-2154:, am_csv.columns.get_loc('거래량')] / 10

ss_csv.iloc[-1418:, ss_csv.columns.get_loc('종가')] = ss_csv.iloc[-1418:, ss_csv.columns.get_loc('종가')] * 50
ss_csv.iloc[-1418:, ss_csv.columns.get_loc('시가')] = ss_csv.iloc[-1418:, ss_csv.columns.get_loc('시가')] * 50
ss_csv.iloc[-1418:, ss_csv.columns.get_loc('고가')] = ss_csv.iloc[-1418:, ss_csv.columns.get_loc('고가')] * 50
ss_csv.iloc[-1418:, ss_csv.columns.get_loc('저가')] = ss_csv.iloc[-1418:, ss_csv.columns.get_loc('저가')] * 50

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
    for i in range(len(dataset) - size + 1):             
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


model = load_model("c:\\_data\\sihum\\02_06_sihum70206_1747_02025-0.0079.hdf5")




results = model.evaluate([X1_test, X2_test,X3_test, X4_test], [y1_test, y2_test])
print("mse : ", results)
y_predict = model.predict([X1_test, X2_test,X3_test, X4_test])


y_predict_ss, y_predict_am = model.predict([X_pre_ss[0].reshape(1,180,15), X_pre_ss[0].reshape(1,180,15), X_pre_am[0].reshape(1,180,15), X_pre_am[0].reshape(1,180,15)])

y1_temp = np.zeros([1,15])
y1_temp[0][0] = y_predict_ss[0]

y2_temp = np.zeros([1,15])
y2_temp[0][0] = y_predict_am[0]

y_predict_ss = mms.inverse_transform(y1_temp)
y_predict_am = mms.inverse_transform(y2_temp)

print("2월 7일의 삼성전자  예측 시가 :", round(y_predict_ss[0][0]/50, 2))
print("2월 7일의 아모레 예측 종가 : ", round(y_predict_am[0][0]/10, 2))

r2_ss = r2_score(y1_test, y_predict[0])
r2_am = r2_score(y2_test, y_predict[1])
print("삼성 R2스코어 : ", r2_ss)
print("아모레 R2스코어 : ", r2_am)

# 2월 7일의 삼성전자  예측 시가 : 74579.55
# 2월 7일의 아모레 예측 종가 :  121115.74
# 삼성 R2스코어 :  0.9999502656193989
# 아모레 R2스코어 :  0.9999767392924293


# 2월 7일의 삼성전자  예측 시가 : 74638.16
# 2월 7일의 아모레 예측 종가 :  120634.91
# 삼성 R2스코어 :  0.9999080855104711
# 아모레 R2스코어 :  0.9999691148238176

# 2월 7일의 삼성전자  예측 시가 : 74470.5667257309
# 2월 7일의 아모레 예측 종가 :  120593.09854358435
# 삼성 R2스코어 :  0.9999216441268735
# 아모레 R2스코어 :  0.9999795028976664
############################################################
# 2월 7일의 삼성전자  예측 시가 : 74393.54664683342
# 2월 7일의 아모레 예측 종가 :  120809.93763357401
# 삼성 R2스코어 :  0.9990050084823417
# 아모레 R2스코어 :  0.9998431363686084
#############################################################