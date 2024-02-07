from sklearn.datasets import load_diabetes
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import time
import numpy as np
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, StandardScaler, RobustScaler

datasets = load_diabetes()
X = datasets.data
y = datasets.target          

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, shuffle=True, random_state=1226)           # 1226 713

mms = MinMaxScaler()
mms.fit(X_train)
X_train = mms.transform(X_train)
X_test = mms.transform(X_test)


model = Sequential()
model.add(Dense(8,input_dim=10))
model.add(Dense(16))
model.add(Dense(24))
model.add(Dense(16))
model.add(Dense(8))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
start_time = time.time()
hist = model.fit(X_train, y_train, epochs=500, batch_size=10, validation_split=0.2)
end_time = time.time()

loss = model.evaluate(X_test, y_test)
y_predict = model.predict(X_test)
r2 = r2_score(y_test, y_predict)
def RMSE(aaa, bbb):
    np.sqrt(mean_squared_error(aaa, bbb))
    return np.sqrt(mean_squared_error(aaa, bbb))
rmse = RMSE(y_test, y_predict)
print("로스 : ", loss)

# print("R2스코어 : ", r2)
# print("걸린시간 : ", round(end_time - start_time, 3), "초")
# print("RMSE : ", rmse)

# MinMaxScaler
# 로스 :  1955.16357421875

# MaxAbsScaler
# 로스 :  2052.78759765625
# StandardScaler
# 로스 :  2062.070556640625

# RobustScaler
# 로스 :  2009.228515625










# MinMaxScaler


# MaxAbsScaler

# StandardScaler


# RobustScaler