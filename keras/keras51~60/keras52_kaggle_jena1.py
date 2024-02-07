import numpy as np
import pandas as pd
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout,Input, Conv2D, MaxPooling2D, Flatten, GlobalAveragePooling2D,concatenate, Reshape, LSTM,GRU,Bidirectional
from keras. callbacks import EarlyStopping, ModelCheckpoint
from keras. utils import to_categorical
from sklearn.model_selection import train_test_split 
from sklearn.metrics import f1_score, accuracy_score,r2_score
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler
import time
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import os
import matplotlib as mpl
import seaborn as sns
plt.rc('font', family = 'Malgun Gothic')

path = "c:\\_data\\kaggle\\jena_climate\\"
path2 = "c:\\_data\\kaggle\\jena_climate\\"

datasets = pd.read_csv(path + "jena_climate_2009_2016.csv", index_col=0) 





# print(np.unique(datasets['p (mbar)'],return_counts=True))   # (array([ 913.6 ,  914.1 ,  917.4 , ..., 1015.29, 1015.3 , 1015.35]), array([1, 1, 2, ..., 2, 1, 1], dtype=int64))
# print(np.min(datasets["p (mbar)"]), np.max(datasets['p (mbar)']))   # 913.6 1015.35

# datasets["p (mbar)"] = np.log(datasets["p (mbar)"])

# # 막대 차트를 그림. x축 인덱스를 생성하고, "p (mbar)"의 값을 y축으로 사용합니다.
# y_values = np.array(datasets["p (mbar)"])

# # 막대 차트를 그림. x축으로는 데이터셋의 길이에 맞는 범위를 사용하고, y축으로는 변환된 "p (mbar)"의 값들을 사용합니다.
# plt.bar(range(len(y_values)), y_values)

# # 차트의 제목과 x, y축 레이블을 설정합니다.
# plt.title('p (mbar)의 제곱근')
# plt.xlabel('인덱스')
# plt.ylabel('p (mbar)의 제곱근 값')

# # 차트를 화면에 표시합니다.
# plt.show()



# print(datasets.info())
 #   Column           Non-Null Count   Dtype
# ---  ------           --------------   -----
#  0   p (mbar)         420551 non-null  float64
#  1   T (degC)         420551 non-null  float64
#  2   Tpot (K)         420551 non-null  float64
#  3   Tdew (degC)      420551 non-null  float64
#  4   rh (%)           420551 non-null  float64
#  5   VPmax (mbar)     420551 non-null  float64
#  6   VPact (mbar)     420551 non-null  float64
#  7   VPdef (mbar)     420551 non-null  float64
#  8   sh (g/kg)        420551 non-null  float64
#  9   H2OC (mmol/mol)  420551 non-null  float64
#  10  rho (g/m**3)     420551 non-null  float64
#  11  wv (m/s)         420551 non-null  float64
#  12  max. wv (m/s)    420551 non-null  float64
#  13  wd (deg)         420551 non-null  float64
# dtypes: float64(14)
# print(datasets.shape)       # (420551, 14)

# y = datasets['T (degC)']

# print(y.shape)      # (420551,)


# print(datasets.isna().sum())
# p (mbar)           0
# T (degC)           0
# Tpot (K)           0
# Tdew (degC)        0
# rh (%)             0
# VPmax (mbar)       0
# VPact (mbar)       0
# VPdef (mbar)       0
# sh (g/kg)          0
# H2OC (mmol/mol)    0
# rho (g/m**3)       0
# wv (m/s)           0
# max. wv (m/s)      0
# wd (deg)           0
# dtype: int64


# size = 15
# target_column = 4
size = 30

def split_Xy(dataset, size, target_column):
    X, y = [], []
    for i in range(len(dataset) - size):             
        X_subset = dataset.iloc[i : (i + size), :]      # X에는 모든 행과, 첫 번째 컬럼부터 target_column 전까지의 컬럼을 포함
        y_subset = dataset.iloc[i + size,  target_column]        # Y에는 모든 행과, target_column 컬럼만 포함 
        X.append(X_subset)
        y.append(y_subset)
    return np.array(X), np.array(y)


X, y = split_Xy(datasets, size, 1)




# print(X.shape)      # (420546, 5, 14)
# print(np.unique(y,return_counts=True))      # (420550, 2)
# print(y.shape)      # (420546,)
X = X.reshape(-1, 30, 14)


mms = MinMaxScaler()

X = mms.fit_transform(X.reshape(-1, 3*14)).reshape(-1, 30, 14)
y = mms.fit_transform(y.reshape(-1, 1))

# np.save(path2 + "keras52_kaggle_jena_save_X.npy", X) 
# np.save(path2 + "keras52_kaggle_jena_save_y.npy", y) 


X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=3, test_size=0.15 )






model = Sequential()
model.add(Bidirectional(LSTM(19, return_sequences=True, activation='relu'), input_shape=(30, 14)))
model.add(Bidirectional(LSTM(9, )))
model.add(Dense(97, activation='swish'))
model.add(Dense(21, activation='swish'))
model.add(Dense(16, activation='swish'))
model.add(Dense(21, activation='swish'))
model.add(Dense(1, activation='swish'))
model.summary()


strat_time = time.time()
model.compile(loss='mse', optimizer='adam', metrics=['mse'])
es = EarlyStopping(monitor='val_loss', mode='min', patience=30, verbose=2, restore_best_weights=True)
model.fit(X_train, y_train, epochs=200, batch_size=1000,verbose=2, validation_split=0.15, callbacks=[es])
end_time = time.time()
# print(X_train, X_test)

# 4. 평가, 예측

results = model.evaluate(X_test, y_test)
y_predict = model.predict(X_test)
r2 = r2_score(y_test, y_predict)
# print(y_test)

print('loss' , results)
print("걸리시간 : ", round(end_time - strat_time, 3), "초")
print("r2_score : ", r2)





# LSTM(bidirectional 씀)
# loss [1.9558448911993764e-05, 1.9558448911993764e-05]
# 걸리시간 :  529.111 초
# r2_score :  0.9989910262603748


# LSTM(bidirectional 안씀)
# loss [3.471592903137207, 0.0005865288549102843]
# 걸리시간 :  3107.885 초
# r2_score :  0.9507299097379145



