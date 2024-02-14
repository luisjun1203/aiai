# keras 18_01 복사
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_absolute_error, accuracy_score
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import OneHotEncoder
from keras.utils import to_categorical
from scipy.special import softmax
from sklearn.svm import LinearSVC

# 1.데이터

datasets = load_iris()

X = datasets.data
y = datasets.target

# 2. 모델구성

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=713, stratify=y)

model=LinearSVC(C=1000000, verbose=1, random_state=3, max_iter=1000)
# C가 크면 training 포인트를 정확히 구분(굴곡지다),  C가 작으면 직선에 가깝다.
model.fit(X_train, y_train)

results = model.score(X_test, y_test)
print("model.score : ", results)

y_predict = model.predict(X_test)
print(y_predict)
#   [1 2 2 2 0 0 0 2 2 0 1 1 2 0 0 2 2 2 2 1 1 0 0 2 1 2 0 2 2 0]         
# acc :  1.0
# acc :  1.0
acc = accuracy_score(y_test, y_predict)
print("acc : ", acc)