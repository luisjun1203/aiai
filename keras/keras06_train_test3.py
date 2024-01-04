import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split



X = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([1,2,3,4,6,5,7,8,9,10])

X_train, X_test , y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=True, random_state=4294967295)        # random_state 고정

print(X_train)
print(y_train)
print(X_test)
print(y_test)

# [검색] train과 test를 섞어서 7:3으로 자를 수 있는 방법을 찾아라!!
# 힌트 : 사이킷런




