from sklearn.datasets import load_diabetes
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import time



datasets = load_diabetes()
X = datasets.data
y = datasets.target

# print(X.shape)      # (442, 10)
# print(y.shape)      # (442, )
# print(datasets.feature_names)   # ['age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6']
# print(datasets.DESCR)           

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
model.fit(X_train, y_train, epochs=50, batch_size=10)
end_time = time.time()

loss = model.evaluate(X_test, y_test)
print("로스 : ", loss)
y_predict = model.predict(X_test)
r2 = r2_score(y_test, y_predict)

print("R2스코어 : ", r2)
print("걸린시간 : ", round(end_time - start_time, 3), "초")
# loss = 'mae' , random_state=713, epochs=500, batch_size=10, test_size=0.15
# 로스 :  40.819854736328125
# R2스코어 :  0.6337180345348047

# loss = 'mse' , random_state=713, epochs=100, batch_size=10, test_size=0.15
# 로스 :  2349.983642578125
# R2스코어 :  0.6459759837533623

# loss = 'mse' , random_state=713, epochs=50, batch_size=10, test_size=0.1
# 로스 :  2337.992431640625
# R2스코어 :  0.6477824706165705


# loss = 'mse' , random_state=1226, epochs=50, batch_size=10, test_size=0.1
# 로스 :  2031.322509765625+
# R2스코어 :  0.7246106597957658