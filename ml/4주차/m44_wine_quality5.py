import numpy as np
import pandas as pd
from xgboost import XGBClassifier,XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error, f1_score
from sklearn.ensemble import RandomForestClassifier




path = "C:\\_data\\dacon\\wine\\"

train_csv = pd.read_csv(path + "train.csv", index_col=0)       
# train_csv.to_csv(path + "train_123_csv", index=False)                                             
test_csv = pd.read_csv(path + "test.csv", index_col=0)

submission_csv = pd.read_csv(path + "sample_submission.csv")

# print(train_csv.info())


# y = pd.get_dummies(y)

# print(y)
# print(y.shape)      #(5497, 7)      # 3,4,5,6,7,8,9

# print(X)

# print(X.shape)          # (5497, 12)
# print(y.shape)          #(5497, 7)
# print(y)
# numeric_cols = train_csv.select_dtypes(include=[np.number])






lae = LabelEncoder()
lae.fit(train_csv['type'])
train_csv['type'] = lae.transform(train_csv['type'])
test_csv['type'] = lae.transform(test_csv['type'])

X= train_csv.drop(['quality'], axis=1)
# print(X)
# print(X.shape)
y = train_csv['quality']

def remove_outlier(dataset:pd.DataFrame):
    for label in dataset:
        data = dataset[label]
        q1 = data.quantile(0.25)
        q3 = data.quantile(0.75)
        iqr = q3-q1
        upbound    = q3 + iqr*1.5
        underbound = q1 - iqr*1.5
        dataset.loc[dataset[label] < underbound, label] = underbound
        dataset.loc[dataset[label] > upbound, label] = upbound
        
    return dataset

# print(train_csv.head(10))
# print(test_csv.head(10))

X = remove_outlier(X)
# print(train_csv.shape,x.shape,sep='\n')
# print(train_csv.max(),train_csv.min())
# print(x.max(),x.min())

X = X.astype(np.float32)
y = y.astype(np.float32)


# print(X.shape, y.shape)
#############################################################
# [실습] y의 클래스를 7개에서 3~5개로 줄여서 성능을 비교
#############################################################
y = y.copy()  # 알아서 참고
# ## 힌트 : for문 돌리자
# def remap_classes(y):
#     y_remap = y.copy()
#     for i in range(len(y_remap)):
#         if y_remap.iloc[i] <= 2:
#             y_remap.iloc[i] = 0  # 낮은 품질
#         elif y_remap.iloc[i] <= 5:
#             y_remap.iloc[i] = 1  # 중간 품질
#         else:
#             y_remap.iloc[i] = 2  # 높은 품질
#     return y_remap

# # y의 클래스를 재매핑
# y_remap = remap_classes(y)


# print(y_remap.value_counts())


# for i, v in enumerate(y):
#     if v <=4:
#         y[i] = 0
#     elif v==5:
#         y[i]=1
#     # elif v==6:
#     #     y[i]=2
#     # elif v==7:
#     #     y[i]=3    
#     # elif v==8:
#     #     y[i]=4
#     else:
#         y[i]=2
        
# print(X.value_counts())
# print(y.value_counts())
# print(X.shape)
# print(y.shape)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=777)      

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

parameters = {
    'n_estimators': 4000,  # 부스팅 라운드의 수/ 디폴트 100/ 1 ~ inf/ 정수
    'learning_rate': 0.05,  # 학습률/ 디폴트 0.3/0~1/
    'max_depth': 13,  # 트리의 최대 깊이/ 디폴트 6/ 0 ~ inf/ 정수
    'min_child_weight': 1,  # 자식에 필요한 모든 관측치에 대한 가중치 합의 최소/ 디폴트 1 / 0~inf
    'gamma': 0.1,  # 리프 노드를 추가적으로 나눌지 결정하기 위한 최소 손실 감소/ 디폴트 0/ 0~ inf
    'subsample': 0.8,  # 각 트리마다의 관측 데이터 샘플링 비율/ 디폴트 1 / 0~1
    'colsample_bytree': 0.8,  # 각 트리 구성에 필요한 컬럼(특성) 샘플링 비율/ 디폴트 1 /9 0~1
    'colsample_bylevel': 0.6, #  디폴트 1 / 0~1
    'colsample_bynode': 0.6, #  디폴트 1 / 0~1
    'reg_alpha' : 0.5,   # 디폴트 0 / 0 ~ inf / L1 절대값 가중치 규제(제한) / alpha
    'reg_lambda' :   0.7,   # 디폴트 1 / 0 ~ inf / L2 제곱 가중치 규제(제한) / lambda
    # 'objective' : ['multi:softmax']
}




model = RandomForestClassifier(random_state=3, n_jobs=-1)
# model.set_params(early_stopping_rounds = 10000)


model.fit(X_train, y_train,
        #   eval_set=[(X_train, y_train), (X_test, y_test)],
            # verbose=1,
            # eval_metric ='mlogloss' 
          )


# 4. 평가, 예측
result = model.score(X_test, y_test)
print("최종점수 : ", result)

y_predict = model.predict(X_test)
f1 = f1_score(y_test, y_predict,average='macro')
acc = accuracy_score(y_test, y_predict)

# print(y_predict.shape)
print("f1_score ", f1)
print("acc_score ", acc)

print(y_predict)


# 최종점수 :  0.6872727272727273
# acc_score  0.6872727272727273
# [1. 2. 1. ... 2. 2. 2.]



# 라벨 변경후
# 최종점수 :  0.8
# f1_score  0.6165890427528863
# acc_score  0.8
# [1. 2. 1. ... 2. 2. 2.]

# 라벨 변경전
# 최종점수 :  0.6836363636363636
# f1_score  0.40321808209292065
# acc_score  0.6836363636363636