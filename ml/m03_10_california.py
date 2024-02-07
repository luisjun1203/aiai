from sklearn.datasets import fetch_california_housing
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
from sklearn.model_selection import train_test_split
import time                     # 시간 알고싶을때
from sklearn.svm import LinearSVR
from sklearn.linear_model import Perceptron, LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

datasets = fetch_california_housing()
X = datasets.data
y = datasets.target

# print(y)
# print(np.unique(y, return_counts=True))




X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, shuffle=True, random_state=42)

# print(X_train)
# print(X_test)

models = [
# LinearSVR(),
Perceptron(),
# LinearRegression(),
# KNeighborsRegressor(),
# DecisionTreeRegressor(),
# RandomForestRegressor()
]

for model in models:
    model_name = model.__class__.__name__
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    r2 = r2_score(y_test, predictions)
    print(f"{model_name} r2: {r2:.4f}")



# epochs=10000, batch_size=80, test_size=0.15, random_state=59
# 로스 :  0.5511764883995056
# R2스코어 :  0.6172541292238007

# mse
# 로스 :  0.6345612406730652
# R2스코어 :  0.5226054954200192
# mae
# 로스 :  0.5319810509681702
# R2스코어 :  0.5684984592213133

# model.score :  0.35430089226834105
# R2스코어 :  0.35430089226834105

# LinearSVR r2: -0.4156
# LinearRegression r2: 0.5904
# KNeighborsRegressor r2: 0.1621
# DecisionTreeRegressor r2: 0.6195
# RandomForestRegressor r2: 0.8067



