import numpy as np
from sklearn .datasets import load_iris
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, KFold, cross_val_score

# 1. 데이터
X, y = load_iris(return_X_y=True)

n_splits= 5
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=123)



# 2.모델
model = SVC(C=100, random_state=123, verbose=1)

# 3. 훈련
scores = cross_val_score(model, X, y, cv=kfold)

print("ACC : ", scores, "\n 평균 ACC : ", round(np.mean(scores), 4))

# 4. 평가, 예측


# ACC :  [0.93333333 1.         0.96666667 0.96666667 0.93333333]
# 평균 ACC :  0.96     








