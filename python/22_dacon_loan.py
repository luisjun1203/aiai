import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from keras.utils import to_categorical

# 1. 데이터
path = "c:\\_data\\dacon\\loan\\"

train_csv = pd.read_csv(path + "train.csv", index_col=0)
test_csv = pd.read_csv(path + "test.csv", index_col=0)
submission_csv = pd.read_csv(path + "sample_submission.csv")

train_csv.iloc[28730, 3] = 'MORTGAGE'

print(train_csv.iloc[28730, 3])




# X = train_csv.drop(['대출등급'], axis =1)
# y = train_csv['대출등급']

# print(train_csv['주택소유상태'].value_counts())





# # Index(['대출금액', '대출기간', '근로기간', '주택소유상태', '연간소득', '부채_대비_소득_비율', '총계좌수', '대출목적',
# #        '최근_2년간_연체_횟수', '총상환원금', '총상환이자', '총연체금액', '연체계좌수', '대출등급'], dtype='object')
# # print(train_csv.info())
# #  0   대출금액          96294 non-null  int64
# #  1   대출기간          96294 non-null  object
# #  2   근로기간          96294 non-null  object
# #  3   주택소유상태        96294 non-null  object
# #  4   연간소득          96294 non-null  int64
# #  5   부채_대비_소득_비율   96294 non-null  float64
# #  6   총계좌수          96294 non-null  int64
# #  7   대출목적          96294 non-null  object
# #  8   최근_2년간_연체_횟수  96294 non-null  int64
# #  9   총상환원금         96294 non-null  int64
# #  10  총상환이자         96294 non-null  float64
# #  11  총연체금액         96294 non-null  float64
# #  12  연체계좌수         96294 non-null  float64
# #  13  대출등급          96294 non-null  object


# lae = LabelEncoder()

# lae_


# train_csv['대출기간'] = train_csv['대출기간'].replace({' 36 months' : 36 , ' 60 months' : 60 }).astype(int)
# test_csv['대출기간'] = test_csv['대출기간'].replace({' 36 months' : 36 , ' 60 months' : 60 }).astype(int)








