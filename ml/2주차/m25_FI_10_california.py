from sklearn.datasets import fetch_california_housing
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
from sklearn.model_selection import train_test_split
import time                     # 시간 알고싶을때
from sklearn.svm import LinearSVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score
import pandas as pd

datasets = fetch_california_housing()
X = datasets.data
y = datasets.target

columns = datasets.feature_names
X = pd.DataFrame(X,columns=columns)


fi_str = "0.49305162 0.06417692 0.04481328 0.0250053  0.02300528 0.14707042\
 0.09172754 0.11114962"


fi_str = fi_str.split()
fi_float = [float(s) for s in fi_str]
# print(fi_float)
fi_list = pd.Series(fi_float)

low_idx_list = fi_list[fi_list <= fi_list.quantile(0.25)].index
# print('low_idx_list',low_idx_list)

low_col_list = [X.columns[index] for index in low_idx_list]
if len(low_col_list) > len(X.columns) * 0.25:
    low_col_list = low_col_list[:int(len(X.columns)*0.25)]
# print('low_col_list',low_col_list)
X.drop(low_col_list,axis=1,inplace=True)
print("after X.shape",X.shape)




X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, shuffle=True, random_state=42)

models = [
DecisionTreeRegressor(),
RandomForestRegressor(),
GradientBoostingRegressor(),
XGBRegressor()
]


for model in models:
    model_name = model.__class__.__name__
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    r2 = r2_score(y_test, predictions)
    print(f"{model_name} accuracy: {r2:.4f}")
    print(model.__class__.__name__, ":", model.feature_importances_)



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

# XGBRegressor accuracy: 0.8313
# XGBRegressor : [0.49305162 0.06417692 0.04481328 0.0250053  0.02300528 0.14707042
#  0.09172754 0.11114962]


# XGBRegressor accuracy: 0.8379
# XGBRegressor : [0.5024577  0.0694152  0.04945359 0.15858172 0.10012263 0.11996917]

