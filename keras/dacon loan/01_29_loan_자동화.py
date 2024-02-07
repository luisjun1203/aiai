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
import os

def save_code_to_file(filename=None):
    if filename is None:
        # 현재 스크립트의 파일명을 가져와서 확장자를 txt로 변경
        filename = os.path.splitext(os.path.basename(__file__))[0] + ".txt"
    else:
        filename = filename + ".txt"
    with open(__file__, "r") as file:
        code = file.read()
    
    with open(filename, "w") as file:
        file.write(code)


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
    X_train, X_test, y_train, y_test = train_test_split(X, y1, test_size=0.15, shuffle=True, random_state=a, stratify=y1)
    start = time.time()

    # X_train = np.asarray(X_train).astype(np.float32)
    # X_test = np.asarray(X_test).astype(np.float32)
    # test_csv = np.asarray(test_csv).astype(np.float32)

    # rbs = RobustScaler(quantile_range=(2,98))
    # rbs.fit(X_train)
    # X_train = rbs.transform(X_train)
    # X_test = rbs.transform(X_test)
    # test_csv = rbs.transform(test_csv)
    
    mms = MinMaxScaler()
    mms.fit(X_train)
    X_train = mms.transform(X_train) 
    X_test = mms.transform(X_test)
    test_csv = mms.transform(test_csv)

   


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

    X_train_dnn2 = X_train.reshape(-1, 13)  
    X_test_dnn2 = X_test.reshape(-1, 13) 
    test_csv_dnn2 = test_csv.reshape(-1, 13)

    input_shape_dnn = (13,)
    dip = Input(shape=input_shape_dnn)
    d1 = Dense(19, activation='swish')(dip)
    d2 = Dense(97, activation='swish')(d1)
    d3 = Dense(9, activation='swish')(d2)
    d4 = Dense(21, activation='swish')(d3)

    dop = Dense(16, activation='swish')(d4)


    input_shape_dnn = (13,)
    dip2 = Input(shape=input_shape_dnn)
    d11 = Dense(19, activation='swish')(dip)
    d22 = Dense(99, activation='swish')(d11)
    d33 = Dense(7, activation='swish')(d22)
    d44 = Dense(13, activation='swish')(d33)
    
    dop2 = Dense(16, activation='swish')(d44)

    combined = concatenate([dop, dop2])



    fl = Dense(21, activation='swish')(combined)
    final_output = Dense(7, activation='softmax')(fl)  

    model = Model(inputs=[dip, dip2], outputs=final_output)

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
    model.fit([X_train_dnn, X_train_dnn2], y_train, epochs=b, batch_size=c, validation_split=0.15, verbose=2, callbacks=[es,mcp])


    end = time.time()

    # print(X_test_cnn.shape)
    # print(X_test_dnn.shape)

    results = model.evaluate([X_test_dnn, X_test_dnn2], y_test)
    print("ACC : ", results[1])

    y_predict = model.predict([X_test_dnn, X_test_dnn2]) 
    y_test = ohe.inverse_transform(y_test)
    y_predict = ohe.inverse_transform(y_predict)


    y_submit = model.predict([test_csv_dnn, test_csv_dnn2])  
    y_submit = ohe.inverse_transform(y_submit)

    y_submit = pd.DataFrame(y_submit)
    submission_csv['대출등급'] = y_submit
    # print(y_submit)

    fs = f1_score(y_test, y_predict, average='macro')
    print("f1_score : ", fs)
    print("걸린시간 : ",round(end - start, 3), "초")
    # submission_csv.to_csv(path + "submission_0129_789_.csv", index=False)
    # print(y_submit)
    if fs > 0.92:
        filename = "".join(["..//_data//_save//dacon_loan_3//dacon_loan_3_auto_", "rs_",str(a), "_bs_", str(c),"_f1_", str(fs.round(4))])
        model.save(filename + ".h5")
        submission_csv.to_csv(path + "submisson_02_05_7_mms_auto.csv", index=False)
        save_code_to_file(filename)
    return fs 




    
import random
# for i in range(100000000):
while True:
    # a = random.randrange(1, 4000000000)     # random_state
    a = 2623144940
    b = random.randrange(3000, 5000)           # epochs
    c = random.randrange(1480, 1520)          # patience
    fs = auto(a, b, c, 500, test_csv)
    # print("random_state : ", a)
    if fs > 0.94:
        # print("random_state : ", a)
        print("epochs : ", b)
        print("batch_size : ", c)
        print("f1 : ", fs)
        break











    







