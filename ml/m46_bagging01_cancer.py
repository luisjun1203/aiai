import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.linear_model import LogisticRegression
import warnings

warnings.filterwarnings('ignore')

# 1. 데이터
X, y = load_breast_cancer(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=777, train_size=0.8, stratify=y)

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# parameters = {
    
    
# }
parameters = {
    # 'objective': 'binary:logistic',  # 분류 문제인 경우 이진 분류를 위해 'binary:logistic'으로 설정합니다.
    # 'eval_metric': 'logloss',  # 모델 평가 지표로 로그 손실을 사용합니다.
    'max_depth': 6,  # 트리의 최대 깊이를 설정합니다.
    'learning_rate': 0.1,  # 학습률을 설정합니다.
    'n_estimators': 100,  # 트리의 개수를 설정합니다.
    'subsample': 0.8,  # 각 트리마다 사용될 샘플의 비율을 설정합니다.
    'colsample_bytree': 0.8,  # 각 트리마다 사용될 피처의 비율을 설정합니다.
    'reg_alpha': 0,  # L1 정규화 파라미터를 설정합니다.
    'reg_lambda': 1,  # L2 정규화 파라미터를 설정합니다.
    'random_state': 42  # 랜덤 시드를 설정합니다.
}

# 2. 모델
model = BaggingClassifier(LogisticRegression(),
                          n_estimators=10, # 디폴트
                          n_jobs=-2,
                          random_state=777,
                          # bootstrap=True,   # 디폴트 중복을 허용한다
                          bootstrap=False # 중복 허용 X
                                                      
                          )

# 3. 훈련
model.fit(X_train, y_train)

# 4. 평가, 예측
results = model.score(X_test, y_test)
print("최종 점수 : ", results)

y_predict = model.predict(X_test)
acc = accuracy_score(y_test, y_predict)
print("acc_score : ", acc)

# 최적의 파라미터 :  {'colsample_bytree': 0.6, 'gamma': 0.5, 'learning_rate': 0.1, 'max_depth': 3, 'min_child_weight': 5, 'n_estimators': 300, 'num_class': 30, 'objective': 'multi:softmax', 'subsample': 0.6, 'verbosity': 1}
# best_score :  0.9618073316283036
# model.score :  0.9649122807017544
# accuracy_score :  0.9649122807017544
# 최적튠 ACC :  0.9649122807017544
# 걸린시간 :  123.59 초



# 최적의 파라미터 :  {'colsample_bylevel': 0.6, 'colsample_bynode': 0.6, 'colsample_bytree': 0.6, 'gamma': 1, 'learning_rate': 0.5, 'max_depth': 3, 'min_child_weight': 5, 'n_estimators': 30, 'num_class': 30, 'objective': 'multi:softmax', 'reg_alpha': 0, 'reg_lambda': 1, 'subsample': 0.6, 'verbosity': 1}
# best_score :  0.9765132139812447
# 최종점수 :  0.9517543859649122
# accuracy_score :  0.9517543859649122
# 최적튠 ACC :  0.9517543859649122
# 걸린시간 :  9.76 초


# 최종 점수 :  0.9736842105263158
# acc_score :  0.9736842105263158
