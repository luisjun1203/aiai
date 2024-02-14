# 드롭 아웃 
import warnings
import numpy as np
import pandas as pd
from keras.models import Sequential, Model,load_model
from keras.layers import Dense, Dropout, Input, Conv2D, Flatten, Conv1D
from sklearn.model_selection import train_test_split, KFold, cross_val_score, cross_val_predict, GridSearchCV, RandomizedSearchCV
from sklearn.datasets import load_boston
from sklearn.metrics import r2_score, accuracy_score
from sklearn.preprocessing import OneHotEncoder
import time
from sklearn.utils import all_estimators
import warnings
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import  HalvingGridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings('ignore')


# 1 데이터
datasets = load_boston()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.7, 
    shuffle=True, random_state=4567)

print(x.shape, y.shape) # (506, 13) (506,)

print(x_train.shape, y_train.shape) # (1, 13) (1,)
print(x_test.shape, y_test.shape) # (505, 13) (505,)
# x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1, 1)
# x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1, 1)
print(x_train.shape)    # (354, 13, 1, 1)
print(x_test.shape)     # (152, 13, 1, 1)

print(x_train[0])
print(y_train[0])
# unique, count = np.unique(y_train, return_counts=True)
print(y_train)
# print(unique, count)


print(np.unique(x_train, return_counts=True))
print(y_train.shape, y_test.shape)
# print(pd.value_counts(x_train))
# ohe = OneHotEncoder()
# y_train = ohe.fit_transform(y_train.reshape(-1,1)).toarray()
# y_test = ohe.fit_transform(y_test.reshape(-1,1)).toarray()
print(x_train.shape, y_train.shape) # (354, 13, 1, 1) (354,)

n_splits = 5
# kfold = KFold(n_splits=n_splits, shuffle=True, random_state=123)
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=123)


from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

parameters = [
    {'RF__n_estimators':[100,200], 'RF__max_depth':[6,10,12], 'RF__min_samples_leaf':[3,10]},
    {'RF__max_depth':[6,8,10,12], 'RF__min_samples_leaf':[3,5,7,10]},
    {'RF__min_samples_leaf':[3,5,7,10], 'RF__min_samples_split':[2,3,5,10]},
    {'RF__min_samples_leaf':[3,5,7,10], 'RF__min_samples_split':[2,3,5,10]},
    {'RF__min_samples_split':[2,3,5,10]},
    {'RF__min_samples_split':[2,3,5,10]}
]



# 2 모델
# model = SVC(C=1, kernel='linear', degree=3)
print('==============하빙그리드서치 시작==========================')
pipe = Pipeline([('MM', MinMaxScaler()),
                 ('RF', RandomForestRegressor())])

model = HalvingGridSearchCV(pipe, parameters,
                     cv = kfold,
                     verbose=1,
                    #  refit=True # 디폴트 트루 # 한바퀴 돌린후 다시 돌린다
                     n_jobs=3   # 24개의 코어중 3개 사용 / 전부사용 -1
                     , random_state= 66,
                    # n_iter=10 # 디폴트 10
                    factor=2,
                    min_resources=40)



start_time = time.time()
model.fit(x_train, y_train)
end_time = time.time()

print('최적의 매개변수 : ', model.best_estimator_)
# 최적의 매개변수 :  SVC(C=1, kernel='linear')
print('최적의 파라미터 : ', model.best_params_) # 내가 선택한것
# 최적의 파라미터 :  {'C': 1, 'degree': 3, 'kernel': 'linear'} 우리가 지정한거중에 가장 좋은거
print('best_score : ', model.best_score_)   # 핏한거의 최고의 스코어
# best_score :  0.975
print('model_score : ', model.score(x_test, y_test))    # 
# model_score :  0.9666666666666667


y_predict = model.predict(x_test)
# print('accuracy_score', accuracy_score(y_test, y_predict))

y_pred_best = model.best_estimator_.predict(x_test)
            # SVC(C-1, kernel='linear').predicict(x_test)
# print('최적 튠 ACC : ', accuracy_score(y_test, y_pred_best))

print('걸린신간 : ', round(end_time - start_time, 2), '초')

# import pandas as pd
# print(pd.DataFrame(model.cv_results_).T)