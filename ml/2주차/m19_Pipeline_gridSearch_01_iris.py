### 데이터셋마다 최상의 파라미터를 알고있다.
# 그리드서치, 랜덤서치, 하빙그리드
# 그걸 적용해서 13개 맹그러

from sklearn.datasets import load_iris  # 꽃
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score    # rmse 사용자정의 하기 위해 불러오는것
import time            
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.pipeline import make_pipeline, Pipeline
import numpy as np
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, HalvingGridSearchCV

# 1. 데이터
x , y = load_iris(return_X_y=True)
print(x.shape, y.shape) # (150, 4) (150,)

x_train, x_test, y_train, y_test = train_test_split(x, y, 
            shuffle=True, train_size= 0.7,
            random_state= 7777,
            stratify=y,)    # 스트레티파이 와이(예스)는 분류에서만 쓴다, 트레인 사이즈에 따라 줄여주는것

# scaler = MinMaxScaler()
# x_train = scaler.fit_transform(x_train)
# x_test = scaler.fit_transform(x_test)


parameters = [
    {'RF__n_estimators':[100,200], 'RF__max_depth':[6,10,12], 'RF__min_samples_leaf':[3,10]},
    {'RF__max_depth':[6,8,10,12], 'RF__min_samples_leaf':[3,5,7,10]},
    {'RF__min_samples_leaf':[3,5,7,10], 'RF__min_samples_split':[2,3,5,10]},
    {'RF__min_samples_leaf':[3,5,7,10], 'RF__min_samples_split':[2,3,5,10]},
    {'RF__min_samples_split':[2,3,5,10]},
    {'RF__min_samples_split':[2,3,5,10]}
]


## 2. 모델구성
# model = RandomForestClassifier()
# model = make_pipeline(MinMaxScaler(), RandomForestClassifier())
pipe = Pipeline([('MM', MinMaxScaler()),
                 ('RF', RandomForestClassifier())])

# model = GridSearchCV(pipe, parameters, cv=5, verbose=1)
# model = RandomizedSearchCV(pipe, parameters, cv=5, verbose=1)
model = HalvingGridSearchCV(pipe, parameters, cv=5, verbose=1)

print(np.min(x_train), np.max(x_train)) # 0.1 7.9
print(np.min(x_test), np.max(x_test))   # 0.1 7.7
# 시스템상에서만 스케일러를 먹이고 출력은 되지 않음

# 3 컴파일, 훈련
model.fit(x_train, y_train)


#4. 평가, 예측
results = model.score(x_test, y_test)
print("model.score : ", results)

y_predict = model.predict(x_test)
print(y_predict)
acc = accuracy_score(y_predict, y_test)
print('acc : ', acc)

# acc :  0.9777777777777777