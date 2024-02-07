from sklearn.datasets import load_diabetes
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import time
from sklearn.svm import LinearSVC


datasets = load_diabetes()
X = datasets.data
y = datasets.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, shuffle=True, random_state=1226)           # 1226 713

model=LinearSVC(C=10, verbose=1, random_state=3)



model.fit(X_train, y_train)

result = model.score(X_test, y_test)
print("model.score : ", result)

y_predict = model.predict(X_test)
# print(y_predict)
r2 = r2_score(y_test, y_predict)

print("R2스코어 : ", r2)


# loss = 'mse' , random_state=1226, epochs=50, batch_size=10, test_size=0.1
# 로스 :  2031.322509765625+
# R2스코어 :  0.7246106597957658

# model.score :  0.022222222222222223
# R2스코어 :  0.5699312314693923




