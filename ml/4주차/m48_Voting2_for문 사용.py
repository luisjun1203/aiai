import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, VotingClassifier
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
xgb = XGBClassifier()
rf = RandomForestClassifier()
lr = LogisticRegression()

model = VotingClassifier(
    estimators=[('LR', lr), ('RF', rf), ('xgb', xgb)],
    voting='soft',
    # voting='hard',    # 디폴트
    
)

# 3. 훈련
model.fit(X_train, y_train)

# 4. 평가, 예측
results = model.score(X_test, y_test)
print("최종 점수 : ", results)

y_predict = model.predict(X_test)
acc = accuracy_score(y_test, y_predict)
print("acc_score : ", acc)

model_class = [xgb, rf, lr]

for model2 in model_class:
    model2.fit(X_train, y_train)
    y_pred = model2.predict(X_test)
    score2 = accuracy_score(y_test, y_pred)
    class_name = model2.__class__.__name__
    print("{0} 정확도 : {1:.4f}".format(class_name, score2))




# 최종 점수 :  0.9824561403508771
# acc_score :  0.9824561403508771


# XGBClassifier 정확도 : 0.9912
# RandomForestClassifier 정확도 : 0.9737
# LogisticRegression 정확도 : 0.9737