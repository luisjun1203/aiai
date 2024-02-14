import numpy as np
from sklearn.svm import LinearSVC
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
# 1.데이터

X_data = np.array([[0,0], [0,1], [1,0], [1,1]])
y_data = np.array([0,1,1,0])
print(X_data.shape, y_data.shape)   # (4, 2) (4,)

# 2. 모델구성
model = LinearSVC(C=10000000000)
# model = Perceptron()

# 3. 훈련
model.fit(X_data, y_data)

# 4. 평가, 예측

acc = model.score(X_data, y_data)
print("model.score : ", acc)

y_predict = model.predict(X_data)
acc2 = accuracy_score(y_data, y_predict)
print("accuracy_score : ", acc2)
print(y_data)
print(y_predict)

# model.score :  1.0
# accuracy_score :  1.0
# [0 1 1 1]
# [0 1 1 1]