# restore_best_weights
# save_best_only
# 에 대한 고찰


# keras09_1_boston.py
import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import r2_score, accuracy_score
from sklearn.datasets import load_breast_cancer
from keras.models import Sequential, load_model
from keras.layers import Dense
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, StandardScaler, RobustScaler
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau


datasets = load_breast_cancer()

X = datasets.data
y = datasets.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, shuffle=True, random_state=20)

mms = MinMaxScaler()
mms.fit(X_train)
X_train = mms.transform(X_train)
X_test = mms.transform(X_test)

# # 2. 모델 구성
model = Sequential()
model.add(Dense(19,input_dim=30,activation='relu'))
model.add(Dense(97,activation='relu'))
model.add(Dense(9,activation='relu'))
model.add(Dense(21,activation='relu'))
model.add(Dense(28,activation='relu'))
model.add(Dense(1, activation='sigmoid'))
# model.summary()



# 3.컴파일, 훈련
from keras.optimizers import Adam
learning_rate = 0.0001


model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=learning_rate))                                                             
model.fit(X_train, y_train, epochs=1000, batch_size=15, validation_split=0.1)


# model = load_model("c:\\_data\\_save\\MCP\\keras25_MCP1.hdf5")
print("=======================1.기본출력 ==============================")
loss = model.evaluate(X_test, y_test, verbose=0)

y_predict = model.predict(X_test)
y_predict = np.where(y_predict > 0.5, 1, 0)
acc = accuracy_score(y_test, y_predict)
print("lr : {0}, 로스 : {1}, ACC 스코어 : {2} ".format(learning_rate, loss, acc))

# print("lr : {0}, ACC 스코어 : {1}".format(learning_rate, acc))

# print("==============================================================")
# print(hist.history['val_loss'])
# print("==============================================================")
# lr : 1.0, 로스 : 0.6880419254302979 
#  lr : 0.1, 로스 : 0.7026680111885071 
# lr : 0.01, 로스 : 0.02593502588570118 
# lr : 0.001, 로스 : 0.012448018416762352 
# lr : 0.0001, 로스 : 0.08719496428966522

# lr : 1.0, 로스 : 0.7466728091239929, ACC 스코어 : 0.5697674418604651
# lr : 0.1, 로스 : 0.7027755379676819, ACC 스코어 : 0.5697674418604651
# lr : 0.01, 로스 : 0.05158856883645058, ACC 스코어 : 0.9883720930232558
# lr : 0.001, 로스 : 0.07774711400270462, ACC 스코어 : 0.9767441860465116
# lr : 0.0001, 로스 : 0.0005355008761398494, ACC 스코어 : 1.0



# 로스 :  0.018709704279899597
# R2스코어 : 0.9721036669988306