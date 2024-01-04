import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split            # naming rule : 중간중간 _ , class : 대문자



X = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([1,2,3,4,6,5,7,8,9,10])

# [검색] train과 test를 섞어서 7:3으로 자를 수 있는 방법을 찾아라!!
# 힌트 : 사이킷런
                                                                                                   
X_train, X_test , y_train, y_test = train_test_split(X, y,                                  # random_state 고정, ( ) => 파라미터 = 매개변수
                                                     test_size=0.3,                         # default : 0.75             
                                                    #  train_size=0.6                       # 이런 경우 : 데이터 손실
                                                     shuffle=True,                          # default : True
                                                     random_state=4294967295
                                                     )
        

# print(X_train)
# print(y_train)
# print(X_test)
# print(y_test)

model = Sequential()
model.add(Dense(4, input_dim=1))
model.add(Dense(8))
model.add(Dense(16))
model.add(Dense(12))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, epochs=500, batch_size=1)

loss = model.evaluate(X_test, y_test)
result = model.predict([11000, 7])
print("로스 : ", loss)
print("[11000, 7]의 예측값 ", result)





