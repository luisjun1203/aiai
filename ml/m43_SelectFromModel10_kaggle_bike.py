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
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler, RobustScaler,StandardScaler,MaxAbsScaler
from sklearn.utils import all_estimators
from sklearn.model_selection import StratifiedKFold, cross_val_predict, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
import time
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBClassifier, XGBRegressor

import warnings
warnings.filterwarnings ('ignore')

path = "c:\\_data\\kaggle\\bike\\"

train_csv = pd.read_csv(path + "train.csv", index_col=0)
test_csv = pd.read_csv(path + "test.csv", index_col=0)
submission_csv = pd.read_csv(path + "samplesubmission.csv")

X = train_csv.drop(['casual', 'registered', 'count'], axis=1)
y = train_csv['count']     


X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=777, test_size=0.2)

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

parameters = {
    'n_estimators': 4000,  # 부스팅 라운드의 수/ 디폴트 100/ 1 ~ inf/ 정수
    'learning_rate': 0.05,  # 학습률/ 디폴트 0.3/0~1/
    'max_depth': 8,  # 트리의 최대 깊이/ 디폴트 6/ 0 ~ inf/ 정수
    'min_child_weight': 1,  # 자식에 필요한 모든 관측치에 대한 가중치 합의 최소/ 디폴트 1 / 0~inf
    'gamma': 0.1,  # 리프 노드를 추가적으로 나눌지 결정하기 위한 최소 손실 감소/ 디폴트 0/ 0~ inf
    'subsample': 0.6,  # 각 트리마다의 관측 데이터 샘플링 비율/ 디폴트 1 / 0~1
    'colsample_bytree': 0.6,  # 각 트리 구성에 필요한 컬럼(특성) 샘플링 비율/ 디폴트 1 /9 0~1
    'colsample_bylevel': 0.6, #  디폴트 1 / 0~1
    'colsample_bynode': 0.6, #  디폴트 1 / 0~1
    'reg_alpha' : 0.5,   # 디폴트 0 / 0 ~ inf / L1 절대값 가중치 규제(제한) / alpha
    'reg_lambda' :   0.7,   # 디폴트 1 / 0 ~ inf / L2 제곱 가중치 규제(제한) / lambda
    
}


# 2. 모델 구성



model = XGBRegressor()
model.set_params(early_stopping_rounds = 10, **parameters)



# 3. 훈련
model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_test, y_test)],
        verbose=1,
        eval_metric ='rmse' 
        )


# 4. 평가

result = model.score(X_test, y_test)
print("최종점수 : ", result)

y_predict = model.predict(X_test)
r2 = r2_score(y_test, y_predict)
print("r2_score ", r2)
# importance.append((i, acc))
    
##################################################################
# print(model.feature_importances_)
print("=================================================================")


thresholds = np.sort(model.feature_importances_)    

from sklearn.feature_selection import SelectFromModel
print(thresholds)



for i in thresholds:
    selection = SelectFromModel(model, threshold=i, prefit=False)   # 인스턴스 , model에는 feature_importances_들어가있음
    
    select_X_train = selection.transform(X_train)
    select_X_test = selection.transform(X_test)
    print(i,"변형된 X_train : ", select_X_train.shape, "변형된 X_test : ", select_X_test.shape )

    select_model = XGBRegressor()
    select_model.set_params(**parameters, early_stopping_rounds=10,
                            eval_metric='rmse')
    
    select_model.fit(select_X_train, y_train,
                     eval_set = [(select_X_train, y_train), 
                     (select_X_test, y_test)])
    


select_y_predict = select_model.predict(select_X_test)
score = r2_score(y_test, select_y_predict)
print('Trech=%.3f, n=%d, ACC: %2f%%'%(i, select_X_train.shape[1], score*100))