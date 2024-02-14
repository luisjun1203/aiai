# keras 18_01 복사
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_absolute_error, accuracy_score
from scipy.special import softmax
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import pandas as pd



class CustomXGBClassfier(XGBClassifier):        # XGBClassfier을 상속받겠다.
    def __str__(self):
        return 'XGBClassifier()'                

aaa = CustomXGBClassfier()
# aaa는 인스턴스

# 1.데이터

# X, y = load_iris(return_X_y=True)
# print(X.shape, y.shape) #(150, 4) (150,)

datasets = load_iris()
X = datasets.data
y = datasets.target

df = pd.DataFrame(X, columns = datasets.feature_names)
# print(df)
df['Target(Y)'] = y
# print(df)

print("================================================상관계수 히트맵 ========================================")
# print(df.corr())
#                    sepal length (cm)  sepal width (cm)  petal length (cm)  petal width (cm)  Target(Y)
# sepal length (cm)           1.000000         -0.117570           0.871754          0.817941   0.782561
# sepal width (cm)           -0.117570          1.000000          -0.428440         -0.366126  -0.426658
# petal length (cm)           0.871754         -0.428440           1.000000          0.962865   0.949035
# petal width (cm)            0.817941         -0.366126           0.962865          1.000000   0.956547
# Target(Y)                   0.782561         -0.426658           0.949035          0.956547   1.000000

import seaborn as sns
sns.set(font_scale=1.2)
sns.heatmap(data=df.corr(),
            square=True,
            annot=True,     # 표안에 수치 명시
            cbar=True       # 사이드 바
            )
plt.show()
# matplotlib의 3.7.2에서는 수치가 잘나오나, 3.8.0에서는 수치가 안나온다 그래서 버전 롤백
