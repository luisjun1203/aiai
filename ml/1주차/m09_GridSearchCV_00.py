import numpy as np
from sklearn .datasets import load_iris
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler, RobustScaler,StandardScaler,MaxAbsScaler


# 1. 데이터
X, y = load_iris(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=3, test_size=0.2, stratify=y)

best_score = 0
for gamma in [0.001, 0.01, 0.1, 1, 10, 100]:        #리스트는 이터레이터 형태
    for C in [0.001, 0.01, 0.1, 1, 10, 100]:
        model = SVC(gamma=gamma, C=C)
        model.fit(X_train, y_train)
        
        score = model.score(X_test, y_test)

        if score > best_score:
            best_score = score
            best_parameters = {'C' : C, 'gamma' : gamma}
            
print("최고 점수 : {:.2f}".format(best_score))            
print("최적의 매개변수 : ", best_parameters)

# 최고 점수 : 0.97
# 최적의 매개변수 :  {'C': 100, 'gamma': 0.01}













