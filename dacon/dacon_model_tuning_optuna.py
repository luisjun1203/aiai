import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, MaxAbsScaler
from sklearn.metrics import roc_auc_score
import optuna

path = "c:\\_data\\dacon\\tuning\\"

train_csv = pd.read_csv(path + "train.csv", index_col=0)
submission_csv = pd.read_csv(path + "sample_submission.csv")

X = train_csv.drop(['login'], axis=1)
y = train_csv['login']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 데이터 스케일링
# scaler = RobustScaler() 
# scaler = MinMaxScaler()
# scaler = MaxAbsScaler()
scaler = StandardScaler()


scaler.fit(X_train)
x_train = scaler.transform(X_train)
x_test = scaler.transform(X_test)

# 이전 시험의 최적 성능을 저장하기 위한 변수
best_auc = float('-inf')

def objective(trial):
    global best_auc
    
    # 하이퍼파라미터 탐색 공간 정의
    n_estimators = trial.suggest_int('n_estimators', 100, 500)
    max_depth = trial.suggest_int('max_depth', 1, 21)
    min_samples_split = trial.suggest_int('min_samples_split', 2, 20)
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 20)
    max_features = trial.suggest_categorical('max_features', ['sqrt', 'log2', None])
    bootstrap = trial.suggest_categorical('bootstrap', [True, False])
    ccp_alpha = trial.suggest_float('ccp_alpha', 0.0, 1.0)
    
    # 모델 생성
    model = RandomForestClassifier(
        n_estimators=n_estimators, 
        max_depth=max_depth, 
        min_samples_split=min_samples_split, 
        min_samples_leaf=min_samples_leaf, 
        max_features=max_features, 
        bootstrap=bootstrap,
        ccp_alpha=ccp_alpha,
        random_state=42
    )

    # 모델 학습
    model.fit(X_train, y_train)

    # 검증 세트에서의 AUC 계산
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_pred_proba)
    
    # 이전 시험보다 성능이 좋으면 최적 성능 및 파라미터 업데이트
    if auc > best_auc:
        best_auc = auc
    
    return auc


# Optuna를 사용하여 하이퍼파라미터 최적화
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=1000)

# 최적의 하이퍼파라미터 및 AUC 출력
best_params = study.best_params
best_auc = study.best_value
print('Best parameters:', best_params)
print('Best AUC:', best_auc)

# 찾은 최적의 파라미터들을 제출 양식에 맞게 제출
for param, value in best_params.items():
    if param in submission_csv.columns:
        submission_csv[param] = value

submission_csv.to_csv(path + "sample_submission_03_29_10_.csv", index=False)

# Best parameters: {'n_estimators': 604, 'max_depth': 16, 'min_samples_split': 16, 'min_samples_leaf': 7, 'max_features': 'log2', 'bootstrap': True, 'ccp_alpha': 0.01946773755724655}
# Best AUC: 0.851037851037851

# 03/19
#713 0.791693038
#921 0.7923259494
#1111 0.7935917722
#1211 0.7414161392
#828 0.7988924051

# 03/20
# 1000  0.1     
# 900   0.1     
# 800   0.1     0.8066455696
# 1000  0.15        
# 900   0.15        

# 03/21
#
#
#
#
#

# 03/22
