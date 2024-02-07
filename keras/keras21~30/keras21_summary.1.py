from keras.models import Sequential
from keras.layers import Dense
import numpy as np

# 1. 데이터

X = np.array([1,2,3])                   # x.shape (3, 1)
y = np.array([1,2,3])                   # y.shape (3, 1)

model = Sequential()
model.add(Dense(5, input_shape=(1,)))
model.add(Dense(4))
model.add(Dense(2))
model.add(Dense(1))

model.summary()             # 모델구성 후 확인해보기

#  Layer (type)                Output Shape              Param #
# =================================================================
#  dense (Dense)               (None, 5)                 (5) 10             

#  dense_1 (Dense)             (None, 4)                 (20) 24

#  dense_2 (Dense)             (None, 2)                 (8) 10

#  dense_3 (Dense)             (None, 1)                 (2) 3                    # bias 노드 추가되어있음              



























