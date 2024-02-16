import pandas as pd
import numpy as np

data = pd.DataFrame([[2, np.nan, 6, 8, 10],
                     [2, 4, np.nan, 8, np.nan],
                     [2,4,6,8,10],
                     [np.nan, 4, np.nan, 8, np.nan]
                     ])

data = data.transpose()
data.columns = ['x1', 'x2', 'x3', 'x4']
# print(data)

from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from impyute.imputation.cs import mice

imputer = SimpleImputer(strategy='mean')        # 

data2 = imputer.fit_transform(data)             # default : mean
# print(data2)
# [[ 2.          2.          2.          6.        ]
#  [ 6.5         4.          4.          4.        ]
#  [ 6.          4.66666667  6.          6.        ]
#  [ 8.          8.          8.          8.        ]
#  [10.          4.66666667 10.          6.        ]]
imputer = SimpleImputer(strategy='median')        # 중위
data3 = imputer.fit_transform(data)             # default : mean
# print(data3)

imputer = SimpleImputer(strategy='most_frequent')        #가장 자주나오는 
data4 = imputer.fit_transform(data)             # default : mean
# print(data4)

imputer = SimpleImputer(strategy='constant')        #상수 0 
data5 = imputer.fit_transform(data) 
# print(data5)

imputer = SimpleImputer(strategy='constant', fill_value=777)        #상수 0 
data6 = imputer.fit_transform(data) 
# print(data6)


#########################KNN imputer ##############################
imputer = KNNImputer(n_neighbors=5) # KNN알고리즘으로 결측치 처리
data7 = imputer.fit_transform(data)
# print(data7)

imputer = IterativeImputer()        # 선형회귀 알고리즘 inerpolate와 비슷
data8 = imputer.fit_transform(data)
# print(data8)

aaa = mice(data.values, n=10, seed=777) # 1.26.3에서 mice 오류
print(aaa)                            # 1.22.4에서 mice 정상


# [[ 2.          2.          2.          1.34306569]
#  [ 3.98358972  4.          4.          4.        ]
#  [ 6.          5.95299828  6.          5.15815085]
#  [ 8.          8.          8.          8.        ]
#  [10.          9.81199313 10.          7.72749392]]
# print(np.__version__)
# 1.26.3





