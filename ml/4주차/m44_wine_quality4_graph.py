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

print(train_csv.info())

lae = LabelEncoder()
lae.fit(train_csv['type'])
train_csv['type'] = lae.transform(train_csv['type'])
test_csv['type'] = lae.transform(test_csv['type'])

X= train_csv.drop(['quality'], axis=1)
# print(X)
# print(X.shape)
y = train_csv['quality']-3
# y = pd.get_dummies(y)

# print(y)
# print(y.shape)      #(5497, 7)      # 3,4,5,6,7,8,9

# print(X)

# print(X.shape)          # (5497, 12)
# print(y.shape)          #(5497, 7)
# print(y)
# numeric_cols = train_csv.select_dtypes(include=[np.number])



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

# print(filtered_df)

# # print(X)
# # print(y)
print(X.shape)
print(y.shape)
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




model = RandomForestClassifier(random_state=3)
# model.set_params(early_stopping_rounds = 10)


model.fit(X_train, y_train,
        #   eval_set=[(X_train, y_train), (X_test, y_test)],
            # verbose=1,
            # eval_metric ='mlogloss' 
          )


# 4. 평가, 예측
result = model.score(X_test, y_test)
print("최종점수 : ", result)

y_predict = model.predict(X_test)
acc = accuracy_score(y_test, y_predict)
print("acc_score ", acc)

import matplotlib.pyplot as plt

quality_counts = train_csv.groupby('quality').count()


data_counts = quality_counts['type']

# 바 차트를 그립니다.
plt.figure(figsize=(10, 6))
plt.bar(data_counts.index, data_counts.values, color='skyblue')

# 차트 제목과 축 이름을 설정합니다.
plt.title('Wine Quality Distribution')
plt.xlabel('Quality')
plt.ylabel('Number of Samples')

# 차트를 표시합니다.
plt.show()




