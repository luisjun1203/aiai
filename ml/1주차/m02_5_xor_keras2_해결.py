import numpy as np
from sklearn.svm import LinearSVC
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
from keras.models import Sequential
from keras.layers import Dense







# 1.데이터

X_data = np.array([[0,0], [0,1], [1,0], [1,1]])
y_data = np.array([0,1,1,0])
print(X_data.shape, y_data.shape)   # (4, 2) (4,)

# 2. 모델구성
# model = LinearSVC(C=10000000000)
# model = Perceptron()
model = Sequential()
model.add(Dense(19, input_shape=(2,)))
model.add(Dense(97,activation='relu'))
model.add(Dense(9,activation='relu'))
model.add(Dense(21,activation='relu'))
model.add(Dense(1, activation='sigmoid'))



# 3. 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(X_data, y_data, batch_size=1, epochs=100)

# 4. 평가, 예측

results = model.evaluate(X_data, y_data)
# print("loss : ", results[0])
print("acc : ", results[1])

y_predict = model.predict(X_data)
y_predict = np.round(y_predict).reshape(-1,).astype(int)
acc2 = accuracy_score(y_data, y_predict)
print("accuracy_score : ", acc2)
print(y_data)
print(y_predict)

# model.score :  1.0
# accuracy_score :  1.0
# [0 1 1 1]
# [0 1 1 1]

# accuracy_score :  1.0
# [0 1 1 0]
# [0 1 1 0]
