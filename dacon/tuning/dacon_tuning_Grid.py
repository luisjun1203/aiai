import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import optuna

path = "c:\\_data\\dacon\\tuning\\"

train_csv = pd.read_csv(path + "train.csv", index_col=0)
submission_csv = pd.read_csv(path + "sample_submission.csv")

X = train_csv.drop(['login'], axis=1)
y = train_csv['login']

n_splits= 7
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# 데이터 스케일링
scaler = StandardScaler()
scaler.fit(X_train)
x_train = scaler.transform(X_train)
x_test = scaler.transform(X_test)

parameters = [
    {'n_estimators': [500, 1000, 1500], 'max_depth': [6,10,12],
     'min_samples_leaf' : [3, 10]},
    {'max_depth' : [6, 8, 10, 12], 'max_features' : ['sqrt']},
    {'bootstrap' : [True, False],
     'min_samples_split' : [2, 3, 5, 10]},
    {'ccp_alpha' : [0, 0.3, 0.5, 0.7, 1]},
    {'n_jobs' : [10], 'min_samples_split' : [2, 3, 5, 10]}   
]

    # n_estimators = trial.suggest_int('n_estimators', 100, 1000)
    # max_depth = trial.suggest_int('max_depth', 3, 18)
    # min_samples_split = trial.suggest_int('min_samples_split', 2, 20)
    # min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 20)
    # max_features = trial.suggest_categorical('max_features', ['sqrt', 'log2', None])
    # bootstrap = trial.suggest_categorical('bootstrap', [True, False])
    # ccp_alpha = trial.suggest_float('ccp_alpha', 0.0, 1.0)


model = GridSearchCV(RandomForestClassifier(), parameters, cv=kfold, verbose=1,
                    # refit = True,     # default
                     n_jobs=-1)


model.fit(X_train, y_train)

print("최적의 매개변수 : ", model.best_estimator_)

best_params = model.best_params_
print("최적의 파라미터 : ", model.best_params_)

print('best_score : ', model.best_score_)
print('model.score : ', model.score(X_test, y_test))
y_predict = model.predict(X_test)
y_pred_best = model.best_estimator_.predict(X_test)


for param, value in best_params.items():
    if param in submission_csv.columns:
        submission_csv[param] = value

submission_csv.to_csv(path + "sample_submission_04_02_3_.csv", index=False)



