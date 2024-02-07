import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_log_error, mean_squared_error
import time
from sklearn.svm import LinearSVR
import warnings
from sklearn.utils import all_estimators
warnings.filterwarnings ('ignore')

path = "c:\\_data\\kaggle\\bike\\"

train_csv = pd.read_csv(path + "train.csv", index_col=0)
test_csv = pd.read_csv(path + "test.csv", index_col=0)
submission_csv = pd.read_csv(path + "samplesubmission.csv")

X = train_csv.drop(['casual', 'registered', 'count'], axis=1)
y = train_csv['count']      


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.123, shuffle=True, random_state=6544)

# allAlgorithms = all_estimators(type_filter='classifier')
allAlgorithms = all_estimators(type_filter='regressor')

# print("allAlgorithms : ", allAlgorithms)
# print("모델의 갯수 : ", len(allAlgorithms))     # 분류모델의 갯수 :  41, 회귀모델의 갯수 :  55

for name, algorithm in allAlgorithms:
    try:
        #2. 모델 구성
        model = algorithm()
        #3. 컴파일, 훈련
        model.fit(X_train, y_train)
        acc = model.score(X_test, y_test)
        print(name, '의 정답률은 : ', acc)
    except:
        print(name, ' :은 바보멍충이!!')
        # continue

# y_submit = model.predict(test_csv)
# # print(y_submit.shape)   
# # r2 = r2_score(y_test, y_submit)


# submission_csv['count'] = y_submit                                        
# print("model.score : ", loss)
# submission_csv.to_csv(path + "submission_0207_1_.csv", index=False)

# print("음수갯수 : ",submission_csv[submission_csv['count']<0].count())      # 0보다 작은 조건의 모든 데이터셋을 세줘

# y_predict = model.predict(X_test)                                           #rmse 구하기
# def RMSLE(y_test, y_predict):
#     np.sqrt(mean_squared_log_error(y_test, y_predict))
#     return np.sqrt(mean_squared_log_error(y_test, y_predict))
# rmsle = RMSLE(y_test, y_predict)

# def RMSE(y_test, y_predict):
#     np.sqrt(mean_squared_error(y_test, y_predict))
#     return np.sqrt(mean_squared_error(y_test, y_predict))
# rmse = RMSE(y_test, y_predict)
# # print("256255음수갯수 : ",submission_csv[submission_csv['count']<0].count())      # 0보다 작은 조건의 모든 데이터셋을 세줘

# print("RMSLE : ", rmsle)
# print("RMSE : ", rmse)














