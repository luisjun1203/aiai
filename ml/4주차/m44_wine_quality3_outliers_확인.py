import numpy as np
import pandas as pd
from xgboost import XGBClassifier,XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error, f1_score





path = "C:\\_data\\dacon\\wine\\"

train_csv = pd.read_csv(path + "train.csv", index_col=0)       
# train_csv.to_csv(path + "train_123_csv", index=False)                                             
test_csv = pd.read_csv(path + "test.csv", index_col=0)

submission_csv = pd.read_csv(path + "sample_submission.csv")



lae = LabelEncoder()
lae.fit(train_csv['type'])
train_csv['type'] = lae.transform(train_csv['type'])
test_csv['type'] = lae.transform(test_csv['type'])

aaa= train_csv.drop(['quality'], axis=1)
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
# print(len(outliers_loc))
import matplotlib.pyplot as plt
plt.boxplot(aaa)
plt.show()   




# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=777)      

# scaler = MinMaxScaler()
# X_train = scaler.fit_transform(X_train)
# X_test = scaler.transform(X_test)








