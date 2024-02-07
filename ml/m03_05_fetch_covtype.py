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
from sklearn.svm import LinearSVC
from sklearn.linear_model import Perceptron, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
datasets = fetch_covtype()

X = datasets.data
y = datasets.target


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35, shuffle=True, random_state=3, stratify=y)

# print(X_train.shape, X_test.shape)      # (464809, 54) (116203, 54)
# print(y_train.shape, y_test.shape)      # (464809, 7) (116203, 7)


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


# 로스 :  0.6938604116439819
# ACC :  0.701832115650177
# accuracy_score :  0.7018321385850623

# 로스 :  0.5239526033401489
# ACC :  0.7768474221229553
# accuracy_score :  0.7768474135779627



# model.score :  0.5275601780138182
# accuracy_score :  0.5275601780138182

# LinearSVC accuracy: 0.5446
# Perceptron accuracy: 0.5282
# LogisticRegression accuracy: 0.6206
# KNeighborsClassifier accuracy: 0.9651
# DecisionTreeClassifier accuracy: 0.9331
# RandomForestClassifier accuracy: 0.9508




