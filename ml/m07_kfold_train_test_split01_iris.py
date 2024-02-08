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

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


n_splits= 5
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=123)



# 2.모델
model = SVC(C=100, random_state=123, verbose=1)

# 3. 훈련
scores = cross_val_score(model, X_train, y_train, cv=kfold)

print("ACC : ", scores, "\n 평균 ACC : ", round(np.mean(scores), 4))
# print(y_test)
# 4. 평가, 예측
y_predict = cross_val_predict(model, X_test, y_test ,cv=kfold)



print(y_predict)

acc = accuracy_score(y_test, y_predict)
print("accracy score : ", acc)

# ACC :  [0.95833333 0.95833333 1.         0.91666667 1.        ]
#  평균 ACC :  0.9667
# [2 1 1 0 2 2 2 2 1 1 1 2 1 1 0 2 2 0 0 0 1 0 0 0 2 1 0 0 1 2]
# [2 1 1 0 2 2 2 2 1 1 2 1 1 1 0 2 2 0 0 0 1 0 0 0 2 1 0 0 1 2]
# accracy score :  0.9333333333333333


# ACC :  [0.96666667 1.         0.96666667 1.         0.93333333]
#  평균 ACC :  0.9733






































































































































