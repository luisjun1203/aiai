# 이상치 알아보기

import numpy as np

aaa = np.array([-10, 2,3,4,5,6,7,8,9,10,11,12,50])

def outliers(data_out):
    quartile_1, q2, quartile_3 = np.percentile(data_out, [25, 50, 75])      # 사분위 값 25%, 50%, 75% 계산
    print("1사분위 : ", quartile_1)
    print("q2 : ", q2)
    print("3사분위 : ", quartile_3)
    iqr = quartile_3 -quartile_1                # iqr계산 : quartile_3 - quartile_1
    print("iqr : ", iqr)
    lower_bound = quartile_1 - (iqr * 1.5)      # lower_bound와 upper_bound: 이상치를 식별하기 위한 하한값과 상한값을 계산함.
    upper_bound = quartile_3 + (iqr * 1.5)      # 여기서  나온 하한값보다 작거나 상한값보다 큰 애들은 이상치라 판별
    return np.where((data_out > upper_bound) |  # np.where : 이상치의 위치를 반환, | : 또는 
                    (data_out < lower_bound))
    
outliers_loc = outliers(aaa)
print("이상치의 위치 : ", outliers_loc) 

import matplotlib.pyplot as plt
plt.boxplot(aaa)
plt.show()   
    
# 1사분위 :  4.0
# q2 :  7.0
# 3사분위 :  10.0
# iqr :  6.0
# 이상치의 위치 :  (array([ 0, 12], dtype=int64),)
    
    
    