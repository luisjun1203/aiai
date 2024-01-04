import numpy as np
from keras.models import Sequential
from keras.layers import Dense

x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([1,2,3,4,6,5,7,8,9,10])

# 실습 넘파이 리스트의 슬라이싱!! 7:3으로 잘라!!!
x_train = x[:-3]  # [0:7] == [0:-3] == [ : 7] == [ :-3] 
y_train = y[:7]
'''
a = b       # a라는 변수에 b값을 넣어라
a == b      # a 와 b가 같다
'''
x_test = x[-3:]     # [7:10] == [-3: ] == [-3:10]
y_test = y[7:]

print(x_train)      # [1,2,3,4,5,6,7]
print(y_train)      # [1,2,3,4,5,6,7]
print(x_test)       # [8,9,10]
print(y_test)       # [8,9,10]

'''
범위를 유지하면서 임의로 추출하는게 더 좋다
'''

model = Sequential()
model.add(Dense(4,input_dim=1))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=500, batch_size=1)

loss = model.evaluate(x_test, y_test)
result = model.predict([11000, 7])
print("로스 : ", loss)
print("[11000,7]의 예측값 : ", result)

#로스 :  0.11913623660802841
# [11000,7]의 예측값 :  [[1.0402873e+04]
#  [6.7662935e+00]]
