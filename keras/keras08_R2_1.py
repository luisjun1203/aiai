from keras.models import Sequential
from keras.layers import Dense
import numpy as np
from sklearn.model_selection import train_test_split


X = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20])
y = np.array([1,2,4,3,5,7,9,3,8,12,13,8, 14,15, 9, 6,17,23,21,20])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=True, random_state=103)

model = Sequential()
model.add(Dense(4,input_dim=1))
model.add(Dense(18))
model.add(Dense(6))
model.add(Dense(13))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, epochs=500, batch_size=1)
        # 에포가 500번일때 확인하고
        # 에포가 1일때 확인 // 이건 초기 랜덤 확인.

loss = model.evaluate(X_test, y_test)
print("로스 : ", loss)

y_predict = model.predict(X_test)
result = model.predict(X)

from sklearn.metrics import r2_score        # 평가 지표
r2 = r2_score(y_test, y_predict)
print("R2 스코어 : ", r2 )

import matplotlib.pyplot as plt             # 시각화 한다

plt.scatter(X, y)                           # scatter : 점
# plt.plot(X, result, color='gold')         # plot : 선
plt.scatter(X, result, color='pink')
plt.show()


# random_state=412
# 로스 :  1.3310250043869019
# R2 스코어 :  0.9554675693002495

