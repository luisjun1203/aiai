# keras09_1_boston.py
import warnings
warnings.filterwarnings('ignore')
from sklearn.metrics import r2_score
from sklearn.datasets import load_boston
from keras.models import Sequential, load_model
from keras.layers import Dense
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, StandardScaler, RobustScaler
from keras.callbacks import EarlyStopping, ModelCheckpoint 

datasets = load_boston()

X = datasets.data
y = datasets.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, shuffle=True, random_state=20)

mms = MinMaxScaler()
mms.fit(X_train)
X_train = mms.transform(X_train)
X_test = mms.transform(X_test)

# # 2. 모델 구성
# model = Sequential()
# model.add(Dense(19,input_dim=13,activation='relu'))
# model.add(Dense(97,activation='relu'))
# model.add(Dense(9,activation='relu'))
# model.add(Dense(21,activation='relu'))
# model.add(Dense(28,activation='relu'))
# model.add(Dense(1))
# model.summary()

# mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1, save_best_only=True, filepath="c:\\_data\\_save\\MCP\\keras25_MCP1.hdf5")        
# model.compile(loss='mae', optimizer='adam')                                                             
# es = EarlyStopping(monitor='val_loss', mode='min',patience=300, verbose= 20, restore_best_weights=True) 
# hist = model.fit(X_train, y_train, epochs=1000, batch_size=15, validation_split=0.1, callbacks=[es,mcp])

model = load_model("c:\\_data\\_save\\MCP\\keras25_MCP1.hdf5")

loss = model.evaluate(X_test, y_test)
print("로스 : ", loss)

y_predict = model.predict(X_test)
result = model.predict(X)

r2 = r2_score(y_test, y_predict)
print("R2스코어 :", r2)

print("==============================================================")
# print(hist.history['val_loss'])
print("==============================================================")




