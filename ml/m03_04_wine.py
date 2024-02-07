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
from sklearn.svm import LinearSVC
from sklearn.linear_model import Perceptron, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
datasets = load_wine()

X = datasets.data
y = datasets.target


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, shuffle=True, random_state=713, stratify=y)


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

# 로스 :  0.2964943051338196
# ACC :  0.8888888955116272
# accuracy_score :  0.8888888888888888

# model.score :  0.8888888888888888
# accuracy_score :  0.8888888888888888


# LinearSVC accuracy: 0.7407
# Perceptron accuracy: 0.5185
# LogisticRegression accuracy: 0.9259
# KNeighborsClassifier accuracy: 0.6667
# DecisionTreeClassifier accuracy: 0.7407
# RandomForestClassifier accuracy: 0.9630



























