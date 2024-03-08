import numpy as np
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from keras.models import Sequential
from keras.layers import Dense, Dropout



# . 데이터
datasets = load_wine()

X = datasets.data
y = datasets['target']
# print(X.shape, y.shape)     # (178, 13) (178,)
# print(np.unique(y, return_counts=True))         # (array([0, 1, 2]), array([59, 71, 48], dtype=int64))
# print(pd.value_counts(y))
# 1    71
# 0    59
# 2    48
# Name: count, dtype: int64
# print(y)
X = X[:-35]
y = y[:-35]
# print(X)
# print(y)
# print(np.unique(y, return_counts=True))         # (array([0, 1, 2]), array([59, 71, 13], dtype=int64))






X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, shuffle=True,random_state=3, stratify=y)


#################### smote ##################################
print("========================= smote 적용 =================================================")
from imblearn.over_sampling import SMOTE
# import sklearn as sk
# print("사이킷런 버전 : ", sk.__version__)   # 1.1.3

smote = SMOTE(random_state=713)
X_train, y_train = smote.fit_resample(X_train, y_train)



# print(X_train.shape)               # (180, 13)
# print(pd.value_counts(y_train))
# 0    60
# 1    60
# 2    60


model = Sequential()
model.add(Dense(1997, input_shape=(13,)))
model.add(Dense(3, activation='softmax'))


model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(X_train, y_train, epochs=100, batch_size=16, validation_split=0.15, verbose=2)


# 4. 평가 예측
results = model.evaluate(X_test, y_test)
print("loss : ", results[0])
print("acc : ", results[1])

y_predict = model.predict(X_test)
y_predict = np.argmax(y_predict, axis=1)
fs = f1_score(y_test, y_predict, average='weighted')
acc = accuracy_score(y_test, y_predict)
print(y_predict)
print("f1 : ", fs)
print("정확도 : ", acc)



# [2 2 0 1 1 2 1 1 0 0 1 0 2 1 2 1 0 1 1 0 0 1]
# f1 :  0.8860930735930737
# 정확도 :  0.8636363636363636






