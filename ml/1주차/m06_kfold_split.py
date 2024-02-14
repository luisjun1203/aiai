import numpy as np
from sklearn .datasets import load_iris
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.model_selection import StratifiedKFold
import pandas as pd

# 1. 데이터
# X, y = load_iris(return_X_y=True)
datasets = load_iris()
df = pd.DataFrame(datasets.data, columns = datasets.feature_names)
print(df)   # [150 rows x 4 columns]




n_splits= 3
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=123)

for train_index, val_index in kfold.split(df):
    print("=======================================================================")
    print(train_index, "\n", val_index) 
    print("훈련데이터 갯수 : ", len(train_index),
          "검증데이터 갯수 : ", len(val_index))


'''

# 2.모델
model = SVC(C=100, random_state=123, verbose=1)

# 3. 훈련
scores = cross_val_score(model, X, y, cv=kfold)

print("ACC : ", scores, "\n 평균 ACC : ", round(np.mean(scores), 4))

# 4. 평가, 예측


# ACC :  [0.96666667 1.         0.96666667 1.         0.93333333]
#  평균 ACC :  0.9733

'''






