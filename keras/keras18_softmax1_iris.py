import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from keras.models import Sequential
from keras.layers import Dense, CategoryEncoding
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_absolute_error, accuracy_score
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import OneHotEncoder
from keras.utils import to_categorical

# 1.데이터

datasets = load_iris()
# print(datasets)     #(n, 4)
# print(datasets.DESCR)
# print(datasets.feature_names)                 # ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']

X = datasets.data
y = datasets.target
y = y.reshape(-1, 1)        
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
y = OneHotEncoder(sparse=False).fit_transform(y)
print(y)
print(y.shape)

################ pandas 방식 #######################
# y = pd.get_dummies(y)

# print(y)

################ keras 방식 ####################
# y = to_categorical(y)

# print(y)
# print(y.shape)






# 2. 모델구성

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, shuffle=True, random_state=713)

# model = Sequential()
# model.add(Dense(19, input_dim = 4))
# model.add(Dense(97))
# model.add(Dense(9))
# model.add(Dense(21, activation='sigmoid'))
# model.add(Dense(3, activation='softmax'))


# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics='accuracy')
# es = EarlyStopping(monitor='accuracy', mode='max', patience=30, verbose=3, restore_best_weights=True)
# hist = model.fit(X_train, y_train, epochs=100, batch_size=5, validation_split=0.15, callbacks=[es])

# loss = model.evaluate(X_test, y_test)
# y_predict = model.predict(X_test)               
# y_predict = y_predict.round()
# r2 = r2_score(y_test, y_predict)
# # result = model.predict(X)

# def ACC(aaa, bbb):
#     (accuracy_score(aaa, bbb))
#     return (accuracy_score(aaa, bbb))
# acc = ACC(y_test, y_predict)
# # print(y_predict)
# # print(y_test)


# print("정확도 : ", acc)
# # print("???",result)
# print("로스 : ", loss)
# print("R2 : ", r2)













