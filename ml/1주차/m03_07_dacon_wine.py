# https://dacon.io/competitions/open/235610/mysubmission


import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from keras.utils import to_categorical
from sklearn.svm import LinearSVC
from sklearn.linear_model import Perceptron, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

path = "C:\\_data\\dacon\\wine\\"

train_csv = pd.read_csv(path + "train.csv", index_col=0)       
# train_csv.to_csv(path + "train_123_csv", index=False)                                             
test_csv = pd.read_csv(path + "test.csv", index_col=0)

submission_csv = pd.read_csv(path + "sample_submission.csv")


lae = LabelEncoder()
lae.fit(train_csv['type'])
train_csv['type'] = lae.transform(train_csv['type'])
test_csv['type'] = lae.transform(test_csv['type'])

X = train_csv.drop(['quality'], axis=1)

y = train_csv['quality']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, shuffle=True, random_state=42, stratify=y)       #9266, 781

models = [
LinearSVC(),
Perceptron(),
LogisticRegression(),
KNeighborsClassifier(),
DecisionTreeClassifier(),
RandomForestClassifier()
]

for model in models:
    model_name = model.__class__.__name__
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f"{model_name} accuracy: {accuracy:.4f}")
  

# y_submit = model.predict(test_csv)  
# y_predict = model.predict(X_test) 

# # y_test = np.argmax(y_test, axis=1)
# # y_predict = np.argmax(y_predict, axis=1)
# # y_submit = np.argmax(y_submit, axis=1)+3

# submission_csv['quality'] = y_submit



# submission_csv.to_csv(path + "submission_0207_1_.csv", index=False)

# acc = accuracy_score(y_test, y_predict)
# print("accuracy_score : ", acc)


# model.score :  0.4961346066393815
# accuracy_score :  0.4961346066393815


# LinearSVC accuracy: 0.2101
# Perceptron accuracy: 0.0869
# LogisticRegression accuracy: 0.4679
# KNeighborsClassifier accuracy: 0.4484
# DecisionTreeClassifier accuracy: 0.5471
# RandomForestClassifier accuracy: 0.6376









