from sklearn.datasets import fetch_covtype
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from keras.utils import to_categorical
from sklearn.preprocessing import OneHotEncoder
from keras. callbacks import EarlyStopping

datasets = fetch_covtype()

X = datasets.data
y = datasets.target
# y = y.reshape(-1, 1)
# print(X.shape, y.shape)         # (581012, 54) (581012)
# print(pd.value_counts(y))       # 7


# 2    283301
# 1    211840
# 3     35754
# 7     20510
# 6     17367
# 5      9493
# 4      2747

# 1. scikit-learn 방식
# y = OneHotEncoder(sparse=False).fit_transform(y)
# print(y.shape)      #(581012, 7)
# print(y)            

# 2. pandas 방식
# y = pd.get_dummies(y)
# print(y.shape)       #(581012, 7)


# 3. keras 방식
y = to_categorical(y-1)
# print(y.shape)          # (581012, 7)
# print(y)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=713, stratify=y)

# print(X_train.shape, X_test.shape)      # (464809, 54) (116203, 54)
# print(y_train.shape, y_test.shape)      # (464809, 7) (116203, 7)


model = Sequential()
model.add(Dense(19, input_dim =54))
model.add(Dense(97))
model.add(Dense(9))
model.add(Dense(21))
model.add(Dense(9))
model.add(Dense(21))
model.add(Dense(7, activation='softmax'))


model.compile(loss='categorical_crossentropy', optimizer='adam', metrics='acc')
es = EarlyStopping(monitor='val_loss', mode='min', patience=50, restore_best_weights=True, verbose=1)
hist = model.fit(X_train, y_train, epochs=100, batch_size=5000, validation_split=0.2, callbacks=[es], verbose=1)

results = model.evaluate(X_test, y_test)
print("로스 : ", results[0])
print("ACC : ", results[1])

y_predict = model.predict(X_test)
y_test = np.argmax(y_test, axis=1)
y_predict = np.argmax(y_predict, axis=1)

acc = accuracy_score(y_predict, y_test)
print("accuracy_score : ", acc)


# 로스 :  0.7144363522529602
# ACC :  0.6855589151382446
# accuracy_score :  0.685558892627557












