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

path = "c:\\_data\\kaggle\\jena_climate\\"
path2 = "c:\\_data\\kaggle\\jena_climate\\"

datasets = pd.read_csv(path + "jena_climate_2009_2016.csv", index_col=0) 

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
size = 3

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
X = X.reshape(-1, 3, 14)



# np.save(path2 + "keras52_kaggle_jena_save_X.npy", X) 
# np.save(path2 + "keras52_kaggle_jena_save_y.npy", y) 


X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=3, test_size=0.15 )


X_train = np.asarray(X_train).astype(np.float32)
X_test = np.asarray(X_test).astype(np.float32)




model = Sequential()
model.add(Bidirectional(LSTM(19, return_sequences=True, activation='relu'), input_shape=(3, 14)))
model.add(Bidirectional(LSTM(9, )))
model.add(Dense(97, activation='swish'))
model.add(Dense(21, activation='swish'))
model.add(Dense(16, activation='swish'))
model.add(Dense(21, activation='swish'))
model.add(Dense(1, activation='swish'))
model.summary()


strat_time = time.time()
model.compile(loss='mse', optimizer='adam', metrics=['acc'])
es = EarlyStopping(monitor='val_loss', mode='min', patience=30, verbose=2, restore_best_weights=True)
model.fit(X_train, y_train, epochs=200, batch_size=300,verbose=2, validation_split=0.15, callbacks=[es])
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

