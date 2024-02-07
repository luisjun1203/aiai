


from sklearn.datasets import load_boston
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
from sklearn.model_selection import train_test_split
import time
from sklearn.svm import LinearSVR
import warnings
from sklearn.utils import all_estimators
warnings.filterwarnings ('ignore')

datasets = load_boston()

X = datasets.data
y = datasets.target


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, shuffle=True, random_state=20)

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

