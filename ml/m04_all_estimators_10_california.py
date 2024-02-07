from sklearn.datasets import fetch_california_housing
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
from sklearn.model_selection import train_test_split
import time                     # 시간 알고싶을때
from sklearn.svm import LinearSVR
import warnings
from sklearn.utils import all_estimators
warnings.filterwarnings ('ignore')

datasets = fetch_california_housing()
X = datasets.data
y = datasets.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, shuffle=True, random_state=42)

# print(X_train)
# print(X_test)

# allAlgorithms = all_estimators(type_filter='classifier')
allAlgorithms = all_estimators(type_filter='regressor')

# print("allAlgorithms : ", allAlgorithms)
# print("모델의 갯수 : ", len(allAlgorithms))     # 분류모델의 갯수 :  41, 회귀모델의 갯수 :  55

for name, algorithm in allAlgorithms:
    try:
        #2. 모델 구성
        model = algorithm()
        #3. 컴파일, 훈련
        model.fit(X_train, y_train)
        acc = model.score(X_test, y_test)
        print(name, '의 정답률은 : ', acc)
    except:
        print(name, ' :은 바보멍충이!!')
        # continue


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






