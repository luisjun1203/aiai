import numpy as np
from sklearn.datasets import load_linnerud
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn. multioutput import MultiOutputRegressor



# 1. 데이터

X, y = load_linnerud(return_X_y=True)
# print(X)
# print(y)
# print(X.shape, y.shape)     # (20, 3) (20, 3)

# 최종값  -> X : [  2. 110.  43.], y :  [138.  33.  68.]

# 2. 모델 구성

# model = RandomForestRegressor()
# model = XGBRegressor()        
# model = RandomForestRegressor()
# model = MultiOutputRegressor(CatBoostRegressor())   # 에러
model = CatBoostRegressor(loss_function='MultiRMSE')
# model = MultiOutputRegressor(LGBMRegressor())     # 컬럼이 여러개이면 사용X  방법을 찾아보자
# model = Lasso()
# model = Ridge()
# model = LinearRegression()


model.fit(X, y)

y_predict = model.predict(X)
print(model.__class__.__name__ , '스코어 : ',
      round(mean_absolute_error(y, y_predict), 4))
print(model.predict([[2, 110, 43]]))    # 행렬 형태

# RandomForestRegressor 스코어 :  3.3222
# [[153.8   34.34  64.12]]

# Ridge 스코어 :  7.4569
# [[187.32842123  37.0873515   55.40215097]]

# LinearRegression 스코어 :  7.4567
# [[187.33745435  37.08997099  55.40216714]]

# Lasso 스코어 :  7.4629
# [[186.96683821  36.71930139  55.40868452]]


# XGBRegressor 스코어 :  0.0008
# [[138.0005    33.002136  67.99897 ]]

# LGBM
# ValueError: y should be a 1d array, got an array of shape (20, 3) instead.
# MultiOutputRegressor 스코어 :  8.91
# [[178.6  35.4  56.1]]

# Catboost
# MultiOutputRegressor 스코어 :  0.2154
# [[138.97756017  33.09066774  67.61547996]]


# Catboost (RMSE로 훈련시켜 나온 결과를 mae로 변환)
# CatBoostRegressor 스코어 :  0.0638
# [[138.21649371  32.99740595  67.8741709 ]]