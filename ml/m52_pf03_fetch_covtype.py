from sklearn.datasets import fetch_covtype
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from keras.utils import to_categorical
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures
from keras. callbacks import EarlyStopping
from sklearn.svm import LinearSVC
import warnings
from sklearn.utils import all_estimators
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler, RobustScaler,StandardScaler,MaxAbsScaler, LabelEncoder
from sklearn.utils import all_estimators
import warnings
import time
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier, XGBRFRegressor

warnings.filterwarnings ('ignore')

datasets = fetch_covtype()

X = datasets.data
y = datasets.target

lae = LabelEncoder()
y= lae.fit_transform(y)

pf = PolynomialFeatures(degree=2, include_bias=False)
X_pf = pf.fit_transform(X)


X_train, X_test, y_train, y_test = train_test_split(X_pf, y, random_state=777, test_size=0.2)
# X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=777, test_size=0.2)

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

parameters = {
    'n_estimators': 300,  # 부스팅 라운드의 수/ 디폴트 100/ 1 ~ inf/ 정수
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



model = XGBClassifier()
model.set_params(early_stopping_rounds = 10, **parameters)



# 3. 훈련
model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_test, y_test)],
        verbose=1,
        eval_metric ='mlogloss' 
        )


# 4. 평가

result = model.score(X_test, y_test)
print("최종점수 : ", result)

y_predict = model.predict(X_test)
acc = accuracy_score(y_test, y_predict)
print("acc_score ", acc)
# importance.append((i, acc))
    
# accuracy_score :  0.8838938418643076
# 최적튠 ACC :  0.8838938418643076
    
# 최종점수 :  0.8882645026376256
# acc_score  0.8882645026376256    
    
##################################################################
# print(model.feature_importances_)
# print("=================================================================")


# thresholds = np.sort(model.feature_importances_)    

# from sklearn.feature_selection import SelectFromModel
# print(thresholds)



# for i in thresholds:
#     selection = SelectFromModel(model, threshold=i, prefit=False)   # 인스턴스 , model에는 feature_importances_들어가있음
    
#     select_X_train = selection.transform(X_train)
#     select_X_test = selection.transform(X_test)
#     print(i,"변형된 X_train : ", select_X_train.shape, "변형된 X_test : ", select_X_test.shape )

#     select_model = XGBClassifier()
#     select_model.set_params(**parameters, early_stopping_rounds=10,
#                             eval_metric='mlogloss')
    
#     select_model.fit(select_X_train, y_train,
#                      eval_set = [(select_X_train, y_train), 
#                      (select_X_test, y_test)])
    


# select_y_predict = select_model.predict(select_X_test)
# score = accuracy_score(y_test, select_y_predict)
# print('Trech=%.3f, n=%d, ACC: %2f%%'%(i, select_X_train.shape[1], score*100))

# # Trech=0.106, n=1, ACC: 51.676807%