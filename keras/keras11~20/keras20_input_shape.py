# keras09_1_boston.py

import warnings
warnings.filterwarnings('ignore')


from sklearn.datasets import load_boston
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
from sklearn.model_selection import train_test_split
import time

# 현재 scikit-learn version 1.3.0 보스턴 안됨. 그래서 삭제함
# pip uninstall scikit-learn 
# pip uninstall scikit-image
# pip uninstall scikit-learn-intelex

# pip install scikit-learn==1.1.3

datasets = load_boston()
# print(datasets)

X = datasets.data
y = datasets.target
# print(X)
# print(X.shape)      # (506, 13)     # 열의 개수 알고싶을때
# print(y)
# print(y.shape)      # (506, )

# print(datasets.feature_names)       # 속성 이름 알고싶을때
# ['CRIM' 'ZN' 'INDUS' 'CHAS' 'NOX' 'RM' 'AGE' 'DIS' 'RAD' 'TAX' 'PTRATIO'
#  'B' 'LSTAT']

# print(datasets.DESCR)               # 속성

# [실습]
# train_size 0.7 이상, 0.9이하
# R2 0.8 이상

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, shuffle=True, random_state=20)

model = Sequential()
# model.add(Dense(33,input_dim=13))                 # input_dim :  meteics 형태 차원이 늘어나면 한계가 있음
model.add(Dense(10, input_shape=(13, )))            # input_shape : 행을 제외한 나머지 shape 입력 ex) (10000, 100, 100, 1) -> input_shape = (100, 100, 1)
model.add(Dense(6))
model.add(Dense(21))
model.add(Dense(10))
model.add(Dense(21))
model.add(Dense(1))

model.compile(loss='mae', optimizer='adam')
start_time = time.time()
model.fit(X_train, y_train, epochs=100, batch_size=15)
end_time = time.time()


loss = model.evaluate(X_test, y_test)
print("로스 : ", loss)

y_predict = model.predict(X_test)
result = model.predict(X)

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print("R2스코어 : ", r2)

print("걸린 시간  : ", round(end_time - start_time, 2), "초")





# random_state=20, epochs=1500, batch_size=15
# 로스 :  15.55542278289795
# R2스코어 :  0.80151274445764

