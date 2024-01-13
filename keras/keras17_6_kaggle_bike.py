import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, accuracy_score, mean_squared_log_error
from keras.callbacks import EarlyStopping

path = "c:\\_data\\kaggle\\bike\\"

train_csv = pd.read_csv(path + "train.csv" , index_col=0)
test_csv = pd.read_csv(path + "test.csv", index_col=0)
submission_csv = pd.read_csv(path + "samplesubmission.csv")

X = train_csv.drop(['casual', 'registered', 'count'], axis=1) 
y = train_csv['count']

# print(X.shape)      #(10886, 8)
# print(y.shape)      #(10886)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.385, shuffle=True, random_state=713)

model = Sequential()
model.add(Dense(8,input_dim=8, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(60, activation='relu'))
model.add(Dense(120, activation='relu'))
model.add(Dense(40, activation='relu'))
model.add(Dense(6, activation='relu'))
model.add(Dense(1, activation='relu'))


model.compile(loss='mse', optimizer='adam', metrics='accuracy')
es = EarlyStopping(monitor='val_loss', mode='min', patience=100, restore_best_weights=True)
hist = model.fit(X_train, y_train, epochs= 500, batch_size=700, validation_split=0.15,callbacks=[es])


loss = model.evaluate(X_test, y_test)
y_submit = model.predict(test_csv)
y_predict = model.predict(X_test)
submission_csv['count'] = y_submit
print("mse : ",loss )
submission_csv.to_csv(path + "submission_0110_1.csv", index=False)

print("음수 : ", submission_csv[submission_csv['count']<0].count())

r2 = r2_score(y_test, y_predict)
def RMSLE(y_test, y_predict):
    np.sqrt(mean_squared_log_error(y_test, y_predict))
    return np.sqrt(mean_squared_log_error(y_test, y_predict))
rmsle = RMSLE(y_test, y_predict) 

print("RMSLE : ", rmsle)





