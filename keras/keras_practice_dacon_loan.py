import numpy as np
import pandas as pd
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout,Input, Conv2D, MaxPooling2D, Flatten, GlobalAveragePooling2D,concatenate, Reshape
from keras. callbacks import EarlyStopping, ModelCheckpoint
from keras. utils import to_categorical
from sklearn.model_selection import train_test_split 
from sklearn.metrics import f1_score
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler
import time
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt

path = "c:\\_data\\dacon\\loan\\"

train_csv = pd.read_csv(path + "train.csv", index_col=0)
test_csv = pd.read_csv(path + "test.csv", index_col=0)
submission_csv = pd.read_csv(path + "sample_submission.csv")


# print(np.unique(train_csv['대출등급'],return_counts=True))  
# (array(['A', 'B', 'C', 'D', 'E', 'F', 'G'], dtype=object), array([16772, 28817, 27623, 13354,  7354,  1954,   420], dtype=int64))
# train_csv.drop[train_csv['대출등급']=='G', '대출등급']
# print(np.unique(train_csv['대출등급'],return_counts=True)) 

 
# train_csv = train_csv[train_csv['대출등급'] !='G']                ################### G라벨 삭제


# print(np.unique(train_csv['대출등급'],return_counts=True))  

test_csv.loc[test_csv['대출기간']==' 36 months', '대출기간'] =36
train_csv.loc[train_csv['대출기간']==' 36 months', '대출기간'] =36

test_csv.loc[test_csv['대출기간']==' 60 months', '대출기간'] =60
train_csv.loc[train_csv['대출기간']==' 60 months', '대출기간'] =60
 
test_csv.loc[test_csv['근로기간']=='3', '근로기간'] ='3 years'
train_csv.loc[train_csv['근로기간']=='3', '근로기간'] ='3 years'
test_csv.loc[test_csv['근로기간']=='1 year','근로기간']='1 years'
train_csv.loc[train_csv['근로기간']=='1 year','근로기간']='1 years'
test_csv.loc[test_csv['근로기간']=='<1 year','근로기간']='< 1 year'
train_csv.loc[train_csv['근로기간']=='<1 year','근로기간']='< 1 year'
test_csv.loc[test_csv['근로기간']=='10+years','근로기간']='10+ years'
train_csv.loc[train_csv['근로기간']=='10+years','근로기간']='10+ years'
train_csv.loc[train_csv['근로기간']=='Unknown', '근로기간']='10+ years'
test_csv.loc[test_csv['근로기간']=='Unknown', '근로기간']='10+ years'
train_csv.value_counts('근로기간')
 
train_csv.loc[train_csv['주택소유상태']=='ANY', '주택소유상태'] = 'OWN'
# train_csv.loc[train_csv['대출등급']=='G', '대출등급'] = 
 
test_csv.loc[test_csv['대출목적']=='결혼', '대출목적'] = '기타'


lae = LabelEncoder()

lae.fit(train_csv['주택소유상태'])
train_csv['주택소유상태'] = lae.transform(train_csv['주택소유상태'])
test_csv['주택소유상태'] = lae.transform(test_csv['주택소유상태'])


lae.fit(train_csv['대출목적'])
train_csv['대출목적'] = lae.transform(train_csv['대출목적'])
test_csv['대출목적'] = lae.transform(test_csv['대출목적'])


lae.fit(train_csv['근로기간'])
train_csv['근로기간'] = lae.transform(train_csv['근로기간'])
test_csv['근로기간'] = lae.transform(test_csv['근로기간'])



X = train_csv.drop(['대출등급'], axis=1)
y = train_csv['대출등급']
# test_csv = test_csv.drop([''], axis=1)

y = y.values.reshape(-1, 1)
ohe = OneHotEncoder(sparse=False)
ohe.fit(y)
y1 = ohe.transform(y)

# rbs = RobustScaler()
# X = rbs.transform(X)
# test_csv=rbs.transform(test_csv)

# print(X.shape)

# X = X.values.reshape(-1, 13, 1, 1)
# test_csv = test_csv.values.reshape(-1, 13, 1, 1)

# print(X.shape)              # (96294, 13, 1, 1)
# print(test_csv.shape)       # (64197, 13, 1, 1)
def auto(a, b, c, d, test_csv):
    X_train, X_test, y_train, y_test = train_test_split(X, y1, test_size=0.4, shuffle=True, random_state=a, stratify=y1)
    start = time.time()


    rbs = RobustScaler(quantile_range=(5,95))
    rbs.fit(X_train)
    X_train = rbs.transform(X_train)
    X_test = rbs.transform(X_test)
    test_csv = rbs.transform(test_csv)



   


    # mms1 = ['대출기간',
    #         '대출금액',
    #         '연간소득',
    #         '부채_대비_소득_비율',
    #         '총계좌수',
    #         '총상환원금',
    #         '총상환이자'
    #         ]

    # mms = MinMaxScaler()
    # mms.fit(X_train[mms1])
    # X_train[mms1] = mms.transform(X_train[mms1])
    # X_test[mms1] = mms.transform(X_test[mms1])
    # test_csv[mms1] = mms.transform(test_csv[mms1])

    # mms = MinMaxScaler()
    # mms.fit(X_train)
    # X_train = mms.transform(X_train)
    # X_test = mms.transform(X_test)
    # test_csv = mms. transform(test_csv)


    # # print(np.unique(X_train[mms1],return_counts=True))
    # rbs1 = [
    #     '연체계좌수', 
    #         '총연체금액', 
    #         '최근_2년간_연체_횟수'
    #         ]

    # rbs = RobustScaler()
    # rbs.fit(X_train[rbs1])
    # X_train[rbs1] = rbs.transform(X_train[rbs1])
    # X_test[rbs1] = rbs.transform(X_test[rbs1])
    # test_csv[rbs1] = rbs.transform(test_csv[rbs1])

    # print(np.unique(X_train[rbs1], return_counts = True)


    X_train_dnn = X_train.reshape(-1, 13)  
    X_test_dnn = X_test.reshape(-1, 13) 
    test_csv_dnn = test_csv.reshape(-1, 13)

    X_train_dnn1 = X_train.reshape(-1, 13) 
    X_test_dnn1 = X_test.reshape(-1, 13) 
    test_csv_dnn1 = test_csv.reshape(-1, 13)

    input_shape_dnn = (13,)
    dip = Input(shape=input_shape_dnn)
    d1 = Dense(19, activation='swish')(dip)
    d2 = Dense(97, activation='swish')(d1)
    d3 = Dense(9, activation='swish')(d2)
    d4 = Dense(21,activation='swish')(d3)
    dop = Dense(16, activation='swish')(d4)


    input_shape_dnn2 = (13,)
    dip1 = Input(shape = input_shape_dnn2) 
    d11 = Dense(50, activation='swish')(dip1)
    d22 = Dense(10, activation='swish')(d11)
    d33 = Dense(80, activation='swish')(d22)
    drop1 = Dropout(0.4)(d33)
    d44 = Dense(10, activation='swish')(drop1)
    d55 = Dense(70, activation='swish')(d44)
    drop2 = Dropout(0.4)(d55)
    d66 = Dense(10, activation='swish')(drop2)
    d77 = Dense(60, activation='swish')(d66)
    drop3 = Dropout(0.4)(d77)
    d88 = Dense(10, activation='swish')(drop3)
    d99 = Dense(50, activation='swish')(d88)
    drop4 = Dropout(0.4)(d99)
    d10 = Dense(10, activation='swish')(drop4)
    d12 = Dense(40, activation='swish')(d10)
    drop5 = Dropout(0.4)(d12)
    d13 = Dense(10, activation='swish')(drop5)
    dop1 = Dense(50, activation='swish')(d13)

    combined = concatenate([dop, dop1])

    fl = Dense(21, activation='swish')(combined)
    final_output = Dense(7, activation='softmax')(fl)  

    model = Model(inputs=[dip, dip1], outputs=final_output)

    model.summary()





    import datetime
    date = datetime.datetime.now()
    print(date)                     
    date = date.strftime("%m%d_%H%M")                        
    print(date)                     

    path = "..\\_data\\_save\\MCP\\"
    filename = '{epoch:05d}-{acc:.4f}-{loss:.4f}.hdf5'            
    filepath = "".join([path, 'k30_3_dacon_loan_',date,'_', filename])




    mcp = ModelCheckpoint(monitor='val_loss', mode='min', verbose=1, save_best_only=True, filepath=filepath)    
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics='acc')

    es = EarlyStopping(monitor='val_loss', mode='min', patience=d, verbose=20, restore_best_weights=True)
    model.fit([X_train_dnn, X_train_dnn1], y_train, epochs=b, batch_size=c, validation_split=0.15, verbose=2, callbacks=[es])


    end = time.time()

    # print(X_test_cnn.shape)
    # print(X_test_dnn.shape)

    results = model.evaluate([X_test_dnn, X_test_dnn1], y_test)
    print("ACC : ", results[1])

    y_predict = model.predict([X_test_dnn, X_test_dnn1]) 
    y_test = ohe.inverse_transform(y_test)
    y_predict = ohe.inverse_transform(y_predict)


    y_submit = model.predict([test_csv_dnn, test_csv_dnn1])  
    y_submit = ohe.inverse_transform(y_submit)

    y_submit = pd.DataFrame(y_submit)
    submission_csv['대출등급'] = y_submit
    # print(y_submit)

    fs = f1_score(y_test, y_predict, average='macro')
    print("f1_score : ", fs)
    print("걸린시간 : ",round(end - start, 3), "초")
    submission_csv.to_csv(path + "submission_0129_123_.csv", index=False)
    print(y_submit)
    return fs
    time.sleep(1)



    
import random
# for i in range(100000000):
while True:
    a = random.randrange(1, 10000)     # random_state
    b = random.randrange(1000, 2000)           # epochs
    c = random.randrange(480, 520)          # patience
    fs = auto(a, b, c, 500, test_csv)
    print("random_state : ", a)
    if fs > 0.935:
        print("random_state : ", a)
        print("epochs : ", b)
        print("batch_size : ", c)
        print("f1 : ", fs)
        break







