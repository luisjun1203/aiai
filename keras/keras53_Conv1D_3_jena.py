import numpy as np
import pandas as pd
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout,Input, Conv2D, MaxPooling2D, Flatten, GlobalAveragePooling2D,concatenate, Reshape, LSTM,GRU,Bidirectional, Conv1D
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
from keras.optimizers import SGD
# plt.rc('font', family = 'Malgun Gothic')

path = "c:\\_data\\kaggle\\jena_climate\\"
path2 = "c:\\_data\\kaggle\\jena_climate\\"

datasets = pd.read_csv(path + "jena_climate_2009_2016.csv", index_col=0) 



size = 720

def split_Xy(dataset, size, target_column):
    X, y = [], []
    for i in range(len(dataset) - size - 144 ):             
        X_subset = dataset.iloc[i : (i + size):3, :]      # X에는 모든 행과, 첫 번째 컬럼부터 target_column 전까지의 컬럼을 포함
        y_subset = dataset.iloc[i + size + 144,  target_column]        # Y에는 모든 행과, target_column 컬럼만 포함 
        X.append(X_subset)
        y.append(y_subset)
    return np.array(X), np.array(y)


X, y= split_Xy(datasets, size, 1)

# X = aaa[:,::3, :]
# y = bbb[:]





# print(X)      
# print(X.shape)        # (419687, 720, 14)
# print(y.shape)        # (419687,)

# X = X.reshape(-1, 72, 140)


mms = MinMaxScaler()

X = mms.fit_transform(X.reshape(-1, 240*14)).reshape(-1, 240, 14)
y = mms.fit_transform(y.reshape(-1, 1))

# print(np.unique(X, return_counts=True))
# print(np.unique(y, return_counts=True))


X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=3, test_size=0.15 )

X_train = np.asarray(X_train).astype(np.float16)
X_test = np.asarray(X_test).astype(np.float16)


model = Sequential()
# model.add(GRU(19, return_sequences=True, activation='relu', input_shape=(72, 140)))
# model.add(GRU(9, ))
model.add(Conv1D(filters=19, kernel_size=3, input_shape=(240,14)))
model.add(Flatten())
model.add(Dense(9, activation='swish'))
model.add(Dense(21, activation='swish'))
model.add(Dense(16, activation='swish'))
model.add(Dense(21, activation='swish'))
model.add(Dense(1, activation='swish'))
model.summary()


strat_time = time.time()
model.compile(loss='mse', optimizer='adam', metrics=['mse'])
es = EarlyStopping(monitor='val_loss', mode='min', patience=40, verbose=2, restore_best_weights=True)
model.fit(X_train, y_train, epochs=5, batch_size=2000,verbose=2, validation_split=0.15, callbacks=[es])
end_time = time.time()
# print(X_train, X_test)

# 4. 평가, 예측

results = model.evaluate(X_test, y_test, batch_size=10)
y_predict = model.predict(X_test, batch_size=10)
r2 = r2_score(y_test, y_predict)
# print(y_test)

print('loss' , results)
print("걸리시간 : ", round(end_time - strat_time, 3), "초")
print("r2_score : ", r2)
