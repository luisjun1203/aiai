import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_absolute_error, accuracy_score
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import OneHotEncoder
from keras.utils import to_categorical
from scipy.special import softmax

# 1.데이터

datasets = load_iris()
# print(datasets)     #(n, 4)
# print(datasets.DESCR)
# print(datasets.feature_names)                 # ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']

X = datasets.data
y = datasets.target
# y = y.reshape(-1, 1)        
# y = y.reshape(50, 3)
# print(y.shape)

# (scikit learn 이용할때)
# # # print(X.shape,y.shape)                        # (150, 4) (150, )
# print(y)                                      # y 라벨의 갯수 꼭 확인 (증폭 or 버려)
# print(X)
# # # print(np.unique(y, return_counts=True))       # (array([0, 1, 2]), array([50, 50, 50], dtype=int64))
# # # print(pd.value_counts(y))                       
# # # 0    50   
# # # 1    50
# # # 2    50                                                

################## sklearn 방식 #####################

# one = OneHotEncoder(sparse=False)
# y = one.fit_transform(y)

# y = one.fit_transform(y).toarray()


# y1 = OneHotEncoder(sparse=False).fit_transform(y)            # fit + transform 대신 쓴다
# print(y1)
# print(y.shape)                                              # (150, 3)

################ pandas 방식 #######################
# y = pd.get_dummies(y)                                     

# print(y)
# print(y.shape)                                            # (150, 3)

################ keras 방식 ####################
y = to_categorical(y)

# print(y)
# print(y.shape)                                            # (150, 3)

# 2. 모델구성

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=713, stratify=y)

# print(X_test.shape)
# print(X_train.shape)
# print(y_test.shape)
# print(y_test.shape)

model = Sequential()
model.add(Dense(19, input_dim = 4))
model.add(Dense(97))
model.add(Dense(9))
model.add(Dense(21, activation='sigmoid'))
model.add(Dense(3, activation='softmax'))


model.compile(loss='categorical_crossentropy', optimizer='adam', metrics='accuracy')
es = EarlyStopping(monitor='val_loss', mode='min', patience=30, verbose=3, restore_best_weights=True)
hist = model.fit(X_train, y_train, epochs=100, batch_size=5, validation_split=0.15, callbacks=[es])

results = model.evaluate(X_test, y_test)
print("로스 : ", results[0])
print("ACC : ", results[1])
y_predict = model.predict(X_test)               



# y_test = y_test.reshape(90, )                          #  이런 짓 하지마
# y_predict = y_predict.reshape(-1, )                    #  이런 짓 하지마
# print(y_predict.shape, y_test.shape) # (90,) (90, )    #  이런 짓 하지마

y_test = np.argmax(y_test, axis=1)
y_predict = np.argmax(y_predict, axis=1)
# print(y_test)
# print(y_test.shape, y_predict.shape)

acc = accuracy_score(y_predict, y_test)
print("accuracy_score : ", acc)















