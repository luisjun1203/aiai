import numpy as np
import pandas as pd
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout,Input
from keras. callbacks import EarlyStopping, ModelCheckpoint
from keras. utils import to_categorical
from sklearn.model_selection import train_test_split 
from sklearn.metrics import f1_score
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler
import time

path = "c:\\_data\\dacon\\loan\\"

train_csv = pd.read_csv(path + "train.csv", index_col=0)
test_csv = pd.read_csv(path + "test.csv", index_col=0)
submission_csv = pd.read_csv(path + "sample_submission.csv")


# print(train_csv.shape)          #(96294, 14)
# print(test_csv.shape)           #(64197, 13)
# print(submission_csv.shape)     #(64197, 2)
# print(train_csv.info())
#  #   Column        Non-Null Count  Dtype
# ---  ------        --------------  -----
#  0   대출금액          96294 non-null  int64
#  1   대출기간          96294 non-null  object
#  2   근로기간          96294 non-null  object
#  3   주택소유상태        96294 non-null  object
#  4   연간소득          96294 non-null  int64
#  5   부채_대비_소득_비율   96294 non-null  float64
#  6   총계좌수          96294 non-null  int64
#  7   대출목적          96294 non-null  object
#  8   최근_2년간_연체_횟수  96294 non-null  int64
#  9   총상환원금         96294 non-null  int64
#  10  총상환이자         96294 non-null  float64
#  11  총연체금액         96294 non-null  float64
#  12  연체계좌수         96294 non-null  float64
#  13  대출등급          96294 non-null  object
# dtypes: float64(4), int64(5), object(5)

# print(test_csv.columns)
# Index(['대출금액', '대출기간', '근로기간', '주택소유상태', '연간소득', '부채_대비_소득_비율', '총계좌수', '대출목적',
#        '최근_2년간_연체_횟수', '총상환원금', '총상환이자', '총연체금액', '연체계좌수'], dtype='object')
# print(train_csv.describe())
# print(train_csv.isnull().sum())     # 0
# print(train_csv.isna().sum())       # 0
# print(np.unique(train_csv['대출기간']))     # [' 36 months' ' 60 months']

test_csv.loc[test_csv['대출기간']==' 36 months', '대출기간'] =36
train_csv.loc[train_csv['대출기간']==' 36 months', '대출기간'] =36

test_csv.loc[test_csv['대출기간']==' 60 months', '대출기간'] =60
train_csv.loc[train_csv['대출기간']==' 60 months', '대출기간'] =60

# print(train_csv['대출기간'])
# print(test_csv['대출기간'])

# print(np.unique(train_csv['근로기간'])) 
# ['1 year' '1 years' '10+ years' '10+years' '2 years' '3' '3 years'
#  '4 years' '5 years' '6 years' '7 years' '8 years' '9 years' '< 1 year' '<1 year' 'Unknown']
# print(np.unique(test_csv['근로기간'])) 



test_csv.loc[test_csv['근로기간']=='3', '근로기간'] ='3 years'
train_csv.loc[train_csv['근로기간']=='3', '근로기간'] ='3 years'
test_csv.loc[test_csv['근로기간']=='1 year','근로기간']='1 years'
train_csv.loc[train_csv['근로기간']=='1 year','근로기간']='1 years'
test_csv.loc[test_csv['근로기간']=='<1 year','근로기간']='< 1 year'
train_csv.loc[train_csv['근로기간']=='<1 year','근로기간']='< 1 year'
test_csv.loc[test_csv['근로기간']=='10+years','근로기간']='10+ years'
train_csv.loc[train_csv['근로기간']=='10+years','근로기간']='10+ years'
train_csv.value_counts('근로기간')
# print(np.unique(train_csv['근로기간']))
# ['1 years' '10+ years' '2 years' '3 years' '4 years' '5 years'
#  '6 years' '7 years' '8 years' '9 years' '< 1 year' 'Unknown']

# print(np.unique(test_csv['주택소유상태']))      #['MORTGAGE' 'OWN' 'RENT']
# print(np.unique(train_csv['주택소유상태']))      #['ANY' 'MORTGAGE' 'OWN' 'RENT']
train_csv.loc[train_csv['주택소유상태']=='ANY', '주택소유상태'] = 'OWN'
# print(np.unique(train_csv['주택소유상태']))

# print(np.unique(train_csv['연간소득'],return_counts=True))
# print(np.unique(test_csv['연간소득'],return_counts=True))
# print(pd.value_counts(train_csv['연간소득']))
# print(pd.value_counts(test_csv['연간소득']))

# print(np.unique(train_csv['부채_대비_소득_비율']))
# print(pd.value_counts(train_csv['부채_대비_소득_비율']))

# pd.set_option('display.max_rows', None)
# print(train_csv['대출목적'].value_counts())
# print(test_csv['대출목적'].value_counts())
test_csv.loc[test_csv['대출목적']=='결혼', '대출목적'] = '기타'
# print(test_csv['대출목적'].value_counts())

# print(np.unique(train_csv['총연체금액']))
# print(pd.value_counts(train_csv['총연체금액']))
# 0의 비율이 너무 많아서 칼럼 삭제


# print(np.unique(train_csv['연체계좌수']))
# print(pd.value_counts(train_csv['연체계좌수']))
# 0의 비율이 너무 많아서 칼럼 삭제


lae = LabelEncoder()

lae.fit(train_csv['주택소유상태'])
train_csv['주택소유상태'] = lae.transform(train_csv['주택소유상태'])
test_csv['주택소유상태'] = lae.transform(test_csv['주택소유상태'])

# print(train_csv['주택소유상태'])
# print(test_csv['주택소유상태'])

lae.fit(train_csv['대출목적'])
train_csv['대출목적'] = lae.transform(train_csv['대출목적'])
test_csv['대출목적'] = lae.transform(test_csv['대출목적'])


lae.fit(train_csv['근로기간'])
train_csv['근로기간'] = lae.transform(train_csv['근로기간'])
test_csv['근로기간'] = lae.transform(test_csv['근로기간'])



X = train_csv.drop(['대출등급', '총연체금액', '연체계좌수'], axis=1)
y = train_csv['대출등급']
test_csv = test_csv.drop(['총연체금액', '연체계좌수'], axis=1)


# print(test_csv.columns)
# print(X.shape)      #(96294, 11)
# print(y.shape)      #(96294,)


y = y.values.reshape(-1, 1)
ohe = OneHotEncoder(sparse=False)
ohe.fit(y)
y1 = ohe.transform(y)

# print(y1)
# print(y1.shape)     # (96294, 7)



X_train, X_test, y_train, y_test = train_test_split(X, y1, test_size=0.4, shuffle=True, random_state=3, stratify=y1)


rbs = RobustScaler()
rbs.fit(X_train)
X_train = rbs.transform(X_train)
X_test = rbs.transform(X_test)
test_csv = rbs.transform(test_csv)




i1 = Input(shape = (11,))
d1 = Dense(19,activation='swish')(i1)      
d2 = Dense(97,activation='swish')(d1)
d3 = Dense(9,activation='swish')(d2)
d4 = Dense(21,activation='swish')(d3)
d5 = Dense(16,activation='swish')(d4)
d6 = Dense(21,activation='swish')(d5)
# drop1 = Dropout(0.2)(d6)
o1 = Dense(7,activation='softmax')(d6)
model = Model(inputs = i1, outputs = o1)

import datetime
date = datetime.datetime.now()
print(date)                     
date = date.strftime("%m%d_%H%M")                        
print(date)                     

path = "..\\_data\\_save\\MCP\\"
filename = '{epoch:05d}-{acc:.4f}-{loss:.4f}.hdf5'            
filepath = "".join([path, 'k30_1_dacon_loan_',date,'_', filename])




mcp = ModelCheckpoint(monitor='val_loss', mode='min', verbose=1, save_best_only=True, filepath=filepath)    
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics='acc')
start = time.time()
es = EarlyStopping(monitor='val_loss', mode='min', patience=150, verbose=20, restore_best_weights=True)
model.fit(X_train, y_train, epochs=10000, batch_size=480, validation_split=0.1, verbose=2)


end = time.time()


results = model.evaluate(X_test, y_test)
print("ACC : ", results[1])

y_predict = model.predict(X_test) 
y_test = ohe.inverse_transform(y_test)
y_predict = ohe.inverse_transform(y_predict)


y_submit = model.predict(test_csv)  
y_submit = ohe.inverse_transform(y_submit)

y_submit = pd.DataFrame(y_submit)
submission_csv['대출등급'] = y_submit
print(y_submit)

fs = f1_score(y_test, y_predict, average='weighted')
print("f1_score : ", fs)
print("걸린시간 : ",round(end - start, 3), "초")
submission_csv.to_csv(path + "submission_0122_3_.csv", index=False)










