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

#      x1   x2    x3   x4
# 0   2.0  2.0   2.0  NaN
# 1   NaN  4.0   4.0  4.0
# 2   6.0  NaN   6.0  NaN
# 3   8.0  8.0   8.0  8.0
# 4  10.0  NaN  10.0  NaN

# 결측치 확인
# print(data.isnull())
# print(data.isnull().sum())
# print(data.info())

#  #   Column  Non-Null Count  Dtype
# ---  ------  --------------  -----
#  0   x1      4 non-null      float64
#  1   x2      3 non-null      float64
#  2   x3      5 non-null      float64
#  3   x4      2 non-null      float64

# 1-1. 결측치 삭제
# print(data.dropna())
#     x1   x2   x3   x4
# 3  8.0  8.0  8.0  8.0
# print(data.dropna(axis=1))      # default : axis = 0
#      x3
# 0   2.0
# 1   4.0
# 2   6.0
# 3   8.0
# 4  10.0

# 2-1. 특정값 - 평균
means = data.mean()
# print(means)
data2 = data.fillna(means)
# print(data2)
#      x1        x2    x3   x4
# 0   2.0  2.000000   2.0  6.0
# 1   6.5  4.000000   4.0  4.0
# 2   6.0  4.666667   6.0  6.0
# 3   8.0  8.000000   8.0  8.0
# 4  10.0  4.666667  10.0  6.0

# 2-2 특정값 - 중위
med = data.median()
# print(med)
data3 = data.fillna(med)
# print(data3)
#      x1   x2    x3   x4
# 0   2.0  2.0   2.0  6.0
# 1   7.0  4.0   4.0  4.0
# 2   6.0  4.0   6.0  6.0
# 3   8.0  8.0   8.0  8.0
# 4  10.0  4.0  10.0  6.0

# 2-3 특정값 - 0 채우기
data4 = data.fillna(0)
# print(data4)
#      x1   x2    x3   x4
# 0   2.0  2.0   2.0  0.0
# 1   0.0  4.0   4.0  4.0
# 2   6.0  0.0   6.0  0.0
# 3   8.0  8.0   8.0  8.0
# 4  10.0  0.0  10.0  0.0

# 2-4 특정값 -  ffill
data5 = data.fillna(method='ffill')
data5 = data.ffill()

# print(data5)
#      x1   x2    x3   x4
# 0   2.0  2.0   2.0  NaN
# 1   2.0  4.0   4.0  4.0
# 2   6.0  4.0   6.0  4.0
# 3   8.0  8.0   8.0  8.0
# 4  10.0  8.0  10.0  8.0

# 2-5 특정값 -  bfill
data6 = data.fillna(method='bfill')
data6 = data.bfill()

# print(data6)
#      x1   x2    x3   x4
# 0   2.0  2.0   2.0  4.0
# 1   6.0  4.0   4.0  4.0
# 2   6.0  8.0   6.0  8.0
# 3   8.0  8.0   8.0  8.0
# 4  10.0  NaN  10.0  NaN

############### 특정 컬럼만 #############################

# print(data.loc[data['x1'] >= 4.0])
means = data['x1'].mean()
# print(means)  # 6.5
data['x1'] = data['x1'].fillna(means)
# print(data)

meds = data['x4'].median()
# print(meds)   # 6.0
data['x4'] = data['x4'].fillna(meds)
# print(data)

data['x2'] =data['x2'].fillna(method='ffill') 
print(data)











