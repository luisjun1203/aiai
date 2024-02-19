# 데이터를 하나로 인식하기 때문에 컬럼별로 나눠서 쓰는게 좋을듯

import numpy as np

aaa = np.array([[-10, 2,3,4,5,6,7,8,9,10,11,12,50],
               [100, 200, -30, 400, 500, 600, -70000, 800, 900, 1000, 210, 420, 350]]
               ).T #(13,2)

# print(aaa.shape)

from sklearn.covariance import EllipticEnvelope

outliers = EllipticEnvelope(contamination=.1)      # 데이터포인트의 비율 / 데이터 100개중 contamination가 0.1이면 이상치가 10개

outliers.fit(aaa)
results = outliers.predict(aaa)
print(results)     



