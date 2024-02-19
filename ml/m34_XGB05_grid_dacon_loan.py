import numpy as np
import pandas as pd
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout,Input, Conv2D, MaxPooling2D, Flatten, GlobalAveragePooling2D,concatenate, Reshape
from keras. callbacks import EarlyStopping, ModelCheckpoint
from keras. utils import to_categorical
from sklearn.model_selection import train_test_split 
from sklearn.metrics import f1_score
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler
import time
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import os
from sklearn.svm import LinearSVC
import warnings
from sklearn.utils import all_estimators
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler, RobustScaler,StandardScaler,MaxAbsScaler
from sklearn.utils import all_estimators
import warnings
from sklearn.model_selection import StratifiedKFold, cross_val_predict, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
import time
from xgboost import XGBClassifier, XGBRFRegressor
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import  HalvingGridSearchCV,HalvingRandomSearchCV
from sklearn.pipeline import Pipeline
warnings.filterwarnings ('ignore')



# def save_code_to_file(filename=None):
# if filename is None:
#     # 현재 스크립트의 파일명을 가져와서 확장자를 txt로 변경
#     filename = os.path.splitext(os.path.basename(__file__))[0] + ".txt"
# else:
#     filename = filename + ".txt"
# with open(__file__, "r") as file:
#     code = file.read()

# with open(filename, "w") as file:
#     file.write(code)


path = "c:\\_data\\dacon\\loan\\"

train_csv = pd.read_csv(path + "train.csv", index_col=0)
test_csv = pd.read_csv(path + "test.csv", index_col=0)
submission_csv = pd.read_csv(path + "sample_submission.csv")
  

test_csv.loc[test_csv['대출기간']==' 36 months', '대출기간'] =36
train_csv.loc[train_csv['대출기간']==' 36 months', '대출기간'] =36

test_csv.loc[test_csv['대출기간']==' 60 months', '대출기간'] =60
train_csv.loc[train_csv['대출기간']==' 60 months', '대출기간'] =60

test_csv.loc[test_csv['근로기간']=='3', '근로기간'] ='3 years'
train_csv.loc[train_csv['근로기간']=='3', '근로기간'] ='3 years'
test_csv.loc[test_csv['근로기간']=='1 year','근로기간']='1 years'
train_csv.loc[train_csv['근로기간']=='1 year','근로기간']='1 years'
test_csv.loc[test_csv['근로기간']=='<1 year','근로기간']='< 1 year'
train_csv.loc[train_csv['근로기간']=='<1 year','근로기간']='< 1 year'
test_csv.loc[test_csv['근로기간']=='10+years','근로기간']='10+ years'
train_csv.loc[train_csv['근로기간']=='10+years','근로기간']='10+ years'
train_csv.loc[train_csv['근로기간']=='Unknown', '근로기간']='10+ years'
test_csv.loc[test_csv['근로기간']=='Unknown', '근로기간']='10+ years'
train_csv.value_counts('근로기간')

train_csv.loc[train_csv['주택소유상태']=='ANY', '주택소유상태'] = 'OWN'

test_csv.loc[test_csv['대출목적']=='결혼', '대출목적'] = '기타'

lae = LabelEncoder()

lae.fit(train_csv['주택소유상태'])
train_csv['주택소유상태'] = lae.transform(train_csv['주택소유상태'])
test_csv['주택소유상태'] = lae.transform(test_csv['주택소유상태'])

lae.fit(train_csv['대출목적'])
train_csv['대출목적'] = lae.transform(train_csv['대출목적'])
test_csv['대출목적'] = lae.transform(test_csv['대출목적'])

lae.fit(train_csv['근로기간'])
train_csv['근로기간'] = lae.transform(train_csv['근로기간'])
test_csv['근로기간'] = lae.transform(test_csv['근로기간'])

X = train_csv.drop(['대출등급'], axis=1)
y = train_csv['대출등급']
n_splits= 5
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=123)

y = lae.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, shuffle=True, random_state=42, stratify=y)

mms = MinMaxScaler()
mms.fit(X_train)
X_train = mms.transform(X_train) 
X_test = mms.transform(X_test)
test_csv = mms.transform(test_csv)



parameters = {
    'XGB__n_estimators': [100, 200 ,300],  # 부스팅 라운드의 수
    'XGB__learning_rate': [0.05, 0.1],  # 학습률
    'XGB__max_depth': [3, 6, 9],  # 트리의 최대 깊이
    'XGB__min_child_weight': [1, 5, 10],  # 자식에 필요한 모든 관측치에 대한 가중치 합의 최소
    'XGB__gamma': [0.5, 1, 1.5, 2],  # 리프 노드를 추가적으로 나눌지 결정하기 위한 최소 손실 감소
    'XGB__subsample': [0.6, 0.8, 1.0],  # 각 트리마다의 관측 데이터 샘플링 비율
    'XGB__colsample_bytree': [0.6, 0.8, 1.0],  # 각 트리 구성에 필요한 컬럼(특성) 샘플링 비율
    'XGB__objective': ['multi:softmax'],  # 학습 태스크 파라미터
    'XGB__num_class': [20],  # 분류해야 할 전체 클래스 수, 멀티클래스 분류인 경우 설정
    'XGB__verbosity' : [1] 
}

 #2. 모델 구성
pipe = Pipeline([('SS',StandardScaler()),
                 ('XGB', XGBClassifier(random_state=3608501786))])

model = HalvingGridSearchCV(pipe, parameters,
                     cv = kfold,
                     verbose=1,
                     refit=True,
                     n_jobs=-1,   
                    # n_iter=10 # 디폴트 10
                    factor=2,
                    min_resources=40,
                    random_state=3)



start_time = time.time()
model.fit(X_train, y_train)
end_time = time.time()
print("최적의 매개변수 : ", model.best_estimator_)
# 최적의 매개변수 :  SVC(C=1, kernel='linear')

print("최적의 파라미터 : ", model.best_params_)
# 최적의 파라미터 :  {'C': 1, 'degree': 3, 'kernel': 'linear'}

print('best_score : ', model.best_score_)
print('model.score : ', model.score(X_test, y_test))
# results = model.score(X_test, y_test)
# print(results)
y_predict = model.predict(X_test)
acc = accuracy_score(y_test, y_predict)
print("accuracy_score : ", acc)

y_pred_best = model.best_estimator_.predict(X_test)
print("최적튠 ACC : " , accuracy_score(y_test, y_pred_best))
# best_score :  0.975 
# model.score :  0.9333333333333333
print("걸린시간 : ", round(end_time - start_time, 2), "초")

# 최적의 파라미터 :  {'colsample_bylevel': 0.6, 'colsample_bynode': 0.6, 'colsample_bytree': 0.6, 'gamma': 1, 'learning_rate': 1, 'max_depth': 3, 'min_child_weight': 5, 'n_estimators': 1000, 'num_class': 30, 'objective': 'multi:softmax', 'reg_alpha': 0, 'reg_lambda': 1, 'subsample': 0.6, 'verbosity': 1}
# best_score :  0.7983481843389397
# model.score :  0.8026306680512288
# accuracy_score :  0.8026306680512288
# 최적튠 ACC :  0.8026306680512288
# 걸린시간 :  222.97 초


