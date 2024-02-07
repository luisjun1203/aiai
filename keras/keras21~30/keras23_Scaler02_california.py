from sklearn.datasets import fetch_california_housing
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
from sklearn.model_selection import train_test_split
import time
import numpy as np
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, StandardScaler, RobustScaler

datasets = fetch_california_housing()
X = datasets.data
y = datasets.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, shuffle=True, random_state=33)

mas = MaxAbsScaler()
mas.fit(X_train)
X_train = mas.transform(X_train)
X_test = mas.transform(X_test)



model = Sequential()
model.add(Dense(21,input_dim=8))
model.add(Dense(7))
model.add(Dense(18))
model.add(Dense(12))
model.add(Dense(1))

model.compile(loss='mae', optimizer='adam')                         
# start_time = time.time()
hist = model.fit(X_train, y_train, epochs=1000, batch_size=400, validation_split=0.2)
# end_time = time.time()

loss = model.evaluate(X_test, y_test)
print("로스 : ", loss)

y_predict = model.predict(X_test)
result = model.predict(X)

from sklearn.metrics import r2_score, mean_squared_error
r2 = r2_score(y_test, y_predict)
def RMSE(aaa, bbb):
    np.sqrt(mean_squared_error(aaa, bbb))
    return np.sqrt(mean_squared_error(aaa, bbb))
rmse = RMSE(y_test, y_predict)

print("R2스코어 : ", r2)
# print("걸린시간 : ", round(end_time - start_time, 3), "초")
print("RMSE : ", rmse)




# R2스코어 :  0.5152774806264401
# RMSE :  0.817108307617143


# MinMaxScaler
# R2스코어 :  0.6225164070758967
# RMSE :  0.7210777047658989

# MaxAbsScaler
# R2스코어 :  0.4883265070704771
# RMSE :  0.8395169850717792

# StandardScaler
# R2스코어 :  0.6247123496390798
# RMSE :  0.7189772759959697

# RobustScaler
# R2스코어 :  0.6111911729075133
# RMSE :  0.7318146271633692


