import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error


# 1. 데이터
X,y = load_breast_cancer(return_X_y=True)

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
# importance = [] #  모델이 특정 특성 수로 훈련되었을 때의 정확도(accuracy)를 포함

# for i in range(X.shape[1], 0, -1):  # X.shape[1] : 열의수(특성), (시작, 끝, step)


model = XGBClassifier()
model.set_params(early_stopping_rounds = 10, **parameters)



# 3. 훈련
model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_test, y_test)],
            verbose=1,
            eval_metric ='logloss' 
          )


# 4. 평가, 예측
result = model.score(X_test, y_test)
print("최종점수 : ", result)

y_predict = model.predict(X_test)
acc = accuracy_score(y_test, y_predict)
print("acc_score ", acc)
# importance.append((i, acc))
    
##################################################################
# print(model.feature_importances_)
print("=================================================================")
# print(len(model.feature_importances_))
# for문을 사용해서 피처가 약한놈부터 하나씩 제거
# 30, 29, 28, 27, ...

thresholds = np.sort(model.feature_importances_)    

from sklearn.feature_selection import SelectFromModel
print(thresholds)

# [0.00520446 0.0053471  0.00591745 0.0064075  0.00676363 0.00694486
#  0.00710993 0.00769385 0.00797227 0.00822438 0.00861966 0.00915695
#  0.0099661  0.01028428 0.01194744 0.01722126 0.01748128 0.02006702
#  0.02280043 0.02311854 0.02914348 0.03481073 0.04796001 0.05316569
#  0.07936718 0.0807446  0.08157698 0.1038262  0.11997186 0.15118481]

for i in thresholds:
    selection = SelectFromModel(model, threshold=i, prefit=False)   # 인스턴스 , model에는 feature_importances_들어가있음
    
    select_X_train = selection.transform(X_train)
    select_X_test = selection.transform(X_test)
    print(i,"변형된 X_train : ", select_X_train.shape, "변형된 X_test : ", select_X_test.shape )

    select_model = XGBClassifier()
    select_model.set_params(**parameters, early_stopping_rounds=10,
                            eval_metric='logloss')
    
    select_model.fit(select_X_train, y_train,
                     eval_set = [(select_X_train, y_train), 
                     (select_X_test, y_test)])
    
# result = model.score(X_test, y_test)
# print("최종점수 : ", result)

select_y_predict = select_model.predict(select_X_test)
score = accuracy_score(y_test, select_y_predict)
print('Trech=%.3f, n=%d, ACC: %2f%%'%(i, select_X_train.shape[1], score*100))
# Trech=0.151, n=1, ACC: 86.842105%