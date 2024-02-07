import warnings
warnings.filterwarnings('ignore')


from sklearn.datasets import load_boston
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
from sklearn.model_selection import train_test_split
import time
from sklearn.svm import LinearSVR
from sklearn.linear_model import Perceptron, LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
datasets = load_boston()

X = datasets.data
y = datasets.target


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, shuffle=True, random_state=20)

models = [
LinearSVR(),
Perceptron(),
LinearRegression(),
KNeighborsRegressor(),
DecisionTreeRegressor(),
RandomForestRegressor()
]

for model in models:
    model_name = model.__class__.__name__
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    r2 = r2_score(y_test, predictions)
    print(f"{model_name} r2: {r2:.4f}")





# random_state=42 epochs=1000, batch_size=15, test_size=0.15
# 로스 :  14.557331085205078
# R2스코어 :  0.7770829847720614

# random_state=42 epochs=10000, batch_size=15,test_size=0.15
# 로스 :  14.244630813598633
# R2스코어 :  0.7818713632982545

# random_state=42, epochs=1000, batch_size=15
# 로스 :  13.158198356628418
# R2스코어 :  0.7985079556506842

# random_state=20, epochs=1500, batch_size=15
# 로스 :  15.55542278289795
# R2스코어 :  0.80151274445764

# random_state=20, epochs=1500, batch_size=15
# loss = 'mse'
# R2스코어 :  0.793961027822985
# loss = 'mae'
# R2스코어 :  0.7919380058581664

