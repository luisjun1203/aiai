from sklearn.datasets import fetch_california_housing
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
from sklearn.model_selection import train_test_split
import time
import numpy as np

datasets = fetch_california_housing()
X = datasets.data
y = datasets.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, shuffle=True, random_state=33)

model = Sequential()
model.add(Dense(21,input_dim=8))
model.add(Dense(7))
model.add(Dense(18))
model.add(Dense(12))
model.add(Dense(1))

model.compile(loss='mae', optimizer='adam', metrics='accuracy')   
from keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss', mode='min', patience=100, verbose=2, restore_best_weights=True)                      
start_time = time.time()
hist = model.fit(X_train, y_train, epochs=200, batch_size=400, validation_split=0.2, callbacks=[es])
end_time = time.time()

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
print("걸린시간 : ", round(end_time - start_time, 3), "초")
print("RMSE : ", rmse)