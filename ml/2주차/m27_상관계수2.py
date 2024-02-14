import pandas as pd

df = pd.DataFrame({'A' : [1,2,3,4,5],
                   'B' : [10, 20, 30, 40, 50],
                   'C' : [5, 4, 3, 2, 1],
                   'D' : [3, 7, 5, 1, 4]
                   })
correlations = df.corr()
print(correlations)

#           A         B         C         D
# A  1.000000  1.000000 -1.000000 -0.282843
# B  1.000000  1.000000 -1.000000 -0.282843
# C -1.000000 -1.000000  1.000000  0.282843
# D -0.282843 -0.282843  0.282843  1.000000
