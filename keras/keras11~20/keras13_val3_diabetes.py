from sklearn.datasets import load_diabetes
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import time

datasets = load_diabetes()
X = datasets.data
y = datasets.target          

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, shuffle=True, random_state=1226)           # 1226 713

model = Sequential()
model.add(Dense(8,input_dim=10))
model.add(Dense(16))
model.add(Dense(24))
model.add(Dense(16))
model.add(Dense(8))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
start_time = time.time()
model.fit(X_train, y_train, epochs=100, batch_size=10, validation_split=0.2)
end_time = time.time()

loss = model.evaluate(X_test, y_test)
print("로스 : ", loss)
y_predict = model.predict(X_test)
r2 = r2_score(y_test, y_predict)

print("R2스코어 : ", r2)
print("걸린시간 : ", round(end_time - start_time, 3), "초")



# 로스 :  2165.528076171875
# R2스코어 :  0.7064162150410287

