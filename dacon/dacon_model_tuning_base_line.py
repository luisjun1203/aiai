import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

path = "c:\\_data\\dacon\\tuning\\"

train_csv = pd.read_csv(path + "train.csv", index_col=0)
submission_csv = pd.read_csv(path + "sample_submission.csv")

X_train = train_csv.drop(['login'], axis=1)
y_train = train_csv['login']


parameters = [
    {"n_estimators": [100, 200], "max_depth": [6, 10, 12], "min_samples_leaf": [3, 10]},
    {"max_depth": [6, 8, 10, 12], "min_samples_leaf": [3, 5, 7, 10]},
    {"min_samples_leaf": [3, 5, 7, 10], "min_samples_split": [2, 3, 5, 10]},
    {"min_samples_split": [2, 3, 5, 10]},
]
# RandomForestClassifier 객체 생성
rf = RandomForestClassifier(random_state=43297347)

# GridSearchCV 객체 생성
grid_search = GridSearchCV(estimator=rf, param_grid=parameters, cv=5, n_jobs=-1, verbose=2, scoring='roc_auc')

# GridSearchCV를 사용한 학습
grid_search.fit(X_train, y_train)

# 최적의 파라미터와 최고 점수 출력
best_params = grid_search.best_params_
best_score = grid_search.best_score_

best_params, best_score

submit = pd.read_csv(path + "sample_submission.csv")

# 찾은 최적의 파라미터들을 제출 양식에 맞게 제출
for param, value in best_params.items():
    if param in submit.columns:
        submit[param] = value

submit.to_csv(path + "sample_submission_03_13_base_line.csv", index=False)