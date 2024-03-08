import numpy as np
from sklearn.preprocessing import PolynomialFeatures        # 증폭의 의미 ex) y = wx + b --->  y = (wx + b)**2


X = np.arange(8).reshape(4, 2)
# print(X)

# [[0 1]    -> 0, 0, 1      -> y값
#  [2 3]    -> 4, 6, 9      ->
#  [4 5]    -> 16, 20, 25
#  [6 7]]   -> 36, 42, 49

# pf = PolynomialFeatures(degree=2, include_bias=False) # 차원을 확대하는 개념
# X_pf = pf.fit_transform(X)
# print(X_pf)

# [[ 0.  1.  0.  0.  1.]
#  [ 2.  3.  4.  6.  9.]
#  [ 4.  5. 16. 20. 25.]
#  [ 6.  7. 36. 42. 49.]]

pf = PolynomialFeatures(degree=3
                        , include_bias=False  # default는 True
                        )   # degree : 몇 제곱 할건지 설정 
X_pf = pf.fit_transform(X)
print(X_pf)

print("======================= 컬럼이 3개라면 ================================")


X = np.arange(12).reshape(4, 3)
print(X)
pf = PolynomialFeatures(degree=3
                        , include_bias=False  # default는 True
                        )   # degree : 몇 제곱 할건지 설정 
X_pf = pf.fit_transform(X)
X_pf = X_pf.astype(int)
print(X_pf)

# [[   0    1    2    0    0    0    1    2    4    0    0    0    0    0
#      0    1    2    4    8]
#  [   3    4    5    9   12   15   16   20   25   27   36   45   48   60
#     75   64   80  100  125]
#  [   6    7    8   36   42   48   49   56   64  216  252  288  294  336
#    384  343  392  448  512]
#  [   9   10   11   81   90   99  100  110  121  729  810  891  900  990
#   1089 1000 1100 1210 1331]]



