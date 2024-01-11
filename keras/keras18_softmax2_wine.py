from sklearn.datasets import load_wine
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from keras.utils import to_categorical
from sklearn.preprocessing import OneHotEncoder
from keras. callbacks import EarlyStopping


datasets = load_wine()

X = datasets.data
y = datasets.target
y = y.reshape(-1,1)
# print(X.shape, y.shape) # (178, 13)  (178, )
# print(y)
# print(pd.value_counts(y))       
# 1    71
# 0    59
# 2    48


# 1. scikit-learn 방식
# y = OneHotEncoder(sparse=False).fit_transform(y)
# print(y.shape)
# print(y)            #(178, 3)

# 2. pandas 방식
# y = pd.get_dummies(y)

# 3. keras 방식
y = to_categorical(y)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, shuffle=True, random_state=713, stratify=y)


model = Sequential()
model.add(Dense(19, input_dim=13,activation='sigmoid'))
model.add(Dense(97))
model.add(Dense(9,activation='relu'))
model.add(Dense(21))
model.add(Dense(3, activation='softmax'))


model.compile(loss='categorical_crossentropy', optimizer='adam', metrics='acc')
es = EarlyStopping(monitor='val_loss', mode='min', patience=30, restore_best_weights=True, verbose=1)
hist = model.fit(X_train, y_train, epochs=100, batch_size=5, validation_split=0.2, callbacks=[es], verbose=1)

results = model.evaluate(X_test, y_test)
print("로스 : ", results[0])
print("ACC : ", results[1])

y_predict = model.predict(X_test)
y_test = np.argmax(y_test, axis=1)
y_predict = np.argmax(y_predict, axis=1)

acc = accuracy_score(y_predict, y_test)
print("accuracy_score : ", acc)


# 로스 :  0.2964943051338196
# ACC :  0.8888888955116272
# accuracy_score :  0.8888888888888888

































