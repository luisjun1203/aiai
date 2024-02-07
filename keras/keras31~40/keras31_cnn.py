from tensorflow.python.keras.models import Sequential
# from tensorflow.keras.models import Sequential
# from keras.models import Sequential                       # 똑같음 버전차이
from tensorflow.python.keras.layers import Dense, Conv2D        #  이미지는 2D

model = Sequential()
# model.add(Dense(10, input_shape=(3, 0)))        # 인풋은 (n, 3)
model.add(Conv2D(10, (2, 2), input_shape=(10,10,1)))        # 10 : 다은행으로 넘어갈 output 개수 , (2, 2) :  2X2로 자른 크기 -> kernel_size(가중치 shape) , input_shape = 1 : 흑백, 3 : RGB (가로 세로 색)
model.add(Dense(5))
model.add(Dense(1))














