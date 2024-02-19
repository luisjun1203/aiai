# 이상치 알아보기

import numpy as np

aaa = np.array([-10, 2,3,4,5,6,700,8,9,10,11,12,50])

aaa = aaa.reshape(-1, 1)    #(13, 1)


from sklearn.covariance import EllipticEnvelope

outliers = EllipticEnvelope(contamination=.000000000000000001)      # 데이터포인트의 비율 / 데이터 100개중 contamination가 0.1이면 이상치가 10개

outliers.fit(aaa)
results = outliers.predict(aaa)
print(results)      #[-1  1  1  1  1  1  1  1  1  1  1  1 -1] -1이 이상치

    
    
    