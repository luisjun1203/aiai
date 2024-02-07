from sklearn.datasets import fetch_california_housing
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
from sklearn.model_selection import train_test_split
import time                     # 시간 알고싶을때
from sklearn.svm import LinearSVR


datasets = fetch_california_housing()
X = datasets.data
y = datasets.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, shuffle=True, random_state=42)

# print(X_train)
# print(X_test)

model = LinearSVR(C=10, random_state=921, verbose=1)

model.fit(X_train, y_train)

loss = model.score(X_test, y_test)
print("model.score : ", loss)

y_predict = model.predict(X_test)
result = model.predict(X)

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print("R2스코어 : ", r2)


# epochs=10000, batch_size=80, test_size=0.15, random_state=59
# 로스 :  0.5511764883995056
# R2스코어 :  0.6172541292238007

# mse
# 로스 :  0.6345612406730652
# R2스코어 :  0.5226054954200192
# mae
# 로스 :  0.5319810509681702
# R2스코어 :  0.5684984592213133

# model.score :  0.35430089226834105
# R2스코어 :  0.35430089226834105






