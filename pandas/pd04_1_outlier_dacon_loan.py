import numpy as np
import pandas as pd
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout,Input, Conv2D, MaxPooling2D, GlobalAveragePooling2D, Flatten
from keras. callbacks import EarlyStopping, ModelCheckpoint
from keras. utils import to_categorical
from sklearn.model_selection import train_test_split 
from sklearn.metrics import f1_score
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler
import time
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import  HalvingGridSearchCV, HalvingRandomSearchCV

path = "c:\\_data\\dacon\\loan\\"

train_csv = pd.read_csv(path + "train.csv", index_col=0)
test_csv = pd.read_csv(path + "test.csv", index_col=0)
submission_csv = pd.read_csv(path + "sample_submission.csv")


# print(train_csv.shape)          #(96294, 14)
# print(test_csv.shape)           #(64197, 13)
# print(submission_csv.shape)     #(64197, 2)
# print(train_csv.info())
#  #   Column        Non-Null Count  Dtype
# ---  ------        --------------  -----
#  0   대출금액          96294 non-null  int64
#  1   대출기간          96294 non-null  object
#  2   근로기간          96294 non-null  object
#  3   주택소유상태        96294 non-null  object
#  4   연간소득          96294 non-null  int64
#  5   부채_대비_소득_비율   96294 non-null  float64
#  6   총계좌수          96294 non-null  int64
#  7   대출목적          96294 non-null  object
#  8   최근_2년간_연체_횟수  96294 non-null  int64
#  9   총상환원금         96294 non-null  int64
#  10  총상환이자         96294 non-null  float64
#  11  총연체금액         96294 non-null  float64
#  12  연체계좌수         96294 non-null  float64
#  13  대출등급          96294 non-null  object
# dtypes: float64(4), int64(5), object(5)








# print(test_csv.columns)
# Index(['대출금액', '대출기간', '근로기간', '주택소유상태', '연간소득', '부채_대비_소득_비율', '총계좌수', '대출목적',
#        '최근_2년간_연체_횟수', '총상환원금', '총상환이자', '총연체금액', '연체계좌수'], dtype='object')
# print(train_csv.describe())
# print(train_csv.isnull().sum())     # 0
# print(train_csv.isna().sum())       # 0
# print(np.unique(train_csv['대출기간']))     # [' 36 months' ' 60 months']

test_csv.loc[test_csv['대출기간']==' 36 months', '대출기간'] =36
train_csv.loc[train_csv['대출기간']==' 36 months', '대출기간'] =36

test_csv.loc[test_csv['대출기간']==' 60 months', '대출기간'] =60
train_csv.loc[train_csv['대출기간']==' 60 months', '대출기간'] =60

# print(train_csv['대출기간'])
# print(test_csv['대출기간'])

# print(np.unique(train_csv['근로기간'])) 
# ['1 year' '1 years' '10+ years' '10+years' '2 years' '3' '3 years'
#  '4 years' '5 years' '6 years' '7 years' '8 years' '9 years' '< 1 year' '<1 year' 'Unknown']
# print(np.unique(test_csv['근로기간'])) 



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
# print(np.unique(train_csv['근로기간']))
# ['1 years' '10+ years' '2 years' '3 years' '4 years' '5 years'
#  '6 years' '7 years' '8 years' '9 years' '< 1 year' 'Unknown']

# print(np.unique(test_csv['주택소유상태']))      #['MORTGAGE' 'OWN' 'RENT']
# print(np.unique(train_csv['주택소유상태']))      #['ANY' 'MORTGAGE' 'OWN' 'RENT']
train_csv.loc[train_csv['주택소유상태']=='ANY', '주택소유상태'] = 'OWN'
# print(np.unique(train_csv['주택소유상태']))

# print(np.unique(train_csv['연간소득'],return_counts=True))
# print(np.unique(test_csv['연간소득'],return_counts=True))
# print(pd.value_counts(train_csv['연간소득']))
# print(pd.value_counts(test_csv['연간소득']))

# print(np.unique(train_csv['부채_대비_소득_비율']))
# print(pd.value_counts(train_csv['부채_대비_소득_비율']))

# pd.set_option('display.max_rows', None)
# print(train_csv['대출목적'].value_counts())
# print(test_csv['대출목적'].value_counts())
test_csv.loc[test_csv['대출목적']=='결혼', '대출목적'] = '기타'
# print(test_csv['대출목적'].value_counts())

# print(np.unique(train_csv['총연체금액']))
# print(pd.value_counts(train_csv['총연체금액']))
# 0의 비율이 너무 많아서 칼럼 삭제


# print(np.unique(train_csv['연체계좌수']))
# print(pd.value_counts(train_csv['연체계좌수']))
# 0의 비율이 너무 많아서 칼럼 삭제


lae = LabelEncoder()

lae.fit(train_csv['주택소유상태'])
train_csv['주택소유상태'] = lae.transform(train_csv['주택소유상태'])
test_csv['주택소유상태'] = lae.transform(test_csv['주택소유상태'])

# print(train_csv['주택소유상태'])
# print(test_csv['주택소유상태'])

lae.fit(train_csv['대출목적'])
train_csv['대출목적'] = lae.transform(train_csv['대출목적'])
test_csv['대출목적'] = lae.transform(test_csv['대출목적'])


lae.fit(train_csv['근로기간'])
train_csv['근로기간'] = lae.transform(train_csv['근로기간'])
test_csv['근로기간'] = lae.transform(test_csv['근로기간'])

# print(train_csv.info())

numeric_cols = train_csv.select_dtypes(include=[np.number])


q1 = numeric_cols.quantile(0.25)
q3 = numeric_cols.quantile(0.75)
iqr = q3 - q1

lower_limit = q1 - 1.5*iqr
upper_limit = q3 + 1.5*iqr


for label in numeric_cols:
    lower = lower_limit[label]
    upper = upper_limit[label]
    
    train_csv[label] = np.where(train_csv[label] < lower, lower, train_csv[label])
    train_csv[label] = np.where(train_csv[label] > upper, upper, train_csv[label])



print(train_csv.max())
print(train_csv.min())

X = train_csv.drop(['대출등급', '총연체금액'], axis=1)
y = train_csv['대출등급']
test_csv = test_csv.drop(['총연체금액'], axis=1)

# X = X.values.reshape(96294, 4, 3, 1)
# test_csv = test_csv.values.reshape(64197,4,3,1)

# print(test_csv.columns)
# print(X.shape)      #(96294, 11)
# print(y.shape)      #(96294,)

y = lae.fit_transform(y)

# y = y.values.reshape(-1, 1)
# ohe = OneHotEncoder(sparse=False)
# ohe.fit(y)
# y1 = ohe.transform(y)

# print(y1)
# print(y1.shape)       # (96294, 7)
# print(X.shape)          # (96294, 1)
# print(test_csv.shape)          # (96294, 1)

# X = X.reshape()
# '대출금액', '대출기간', '근로기간', '주택소유상태', '연간소득', '부채_대비_소득_비율', '총계좌수', '대출목적',
# #        '최근_2년간_연체_횟수', '총상환원금', '총상환이자', '총연체금액', '연체계좌수'






#'연간소득'
# 1사분위 :  57600000.0
# q2 :  78000000.0
# 3사분위 :  112800000.0
# iqr :  55200000.0
# 이상치의 위치 :  (array([   34,    50,    78, ..., 96273, 96286, 96289], dtype=int64),)

#'부채_대비_소득_비율'
# 1사분위 :  12.65
# q2 :  18.74
# 3사분위 :  25.54
# iqr :  12.889999999999999
# 이상치의 위치 :  (array([10816, 12498, 15660, 16736, 17895, 19514, 23016, 24930, 26342,
#        26953, 29129, 34936, 39495, 50281, 50868, 51124, 53554, 62125,
#        62765, 69104, 76410, 78932, 81385, 81970, 83954, 85553, 90832,
#        91650, 94676, 95862], dtype=int64),)

# '총계좌수'
# 1사분위 :  17.0
# q2 :  24.0
# 3사분위 :  32.0
# iqr :  15.0
# 이상치의 위치 :  (array([   27,    59,    64, ..., 96155, 96207, 96214], dtype=int64),)

#'총상환원금'
# 1사분위 :  307572.0
# q2 :  597696.0
# 3사분위 :  1055076.0
# iqr :  747504.0
# 이상치의 위치 :  (array([  133,   179,   194, ..., 96179, 96249, 96272], dtype=int64),)




X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, shuffle=True, random_state=3)


splits = 3
kfold = StratifiedKFold(n_splits=splits, shuffle=True, random_state=3)



# X_train = X_train.reshape(-1, 28*28)
# X_test = X_test.reshape(-1, 28*28)



# n_components_list = [154, 331, 486, 713, 784]
# results = []
# for n_components in n_components_list:
#     print(f"n_components: {n_components}")
# lda = LinearDiscriminantAnalysis()
# X_train_pca = lda.fit_transform(X_train, y_train)
# X_test_pca = lda.transform(X_test)

start = time.time()
parameters = {
    'XGB__n_estimators': [100],  # 부스팅 라운드의 수
    'XGB__learning_rate': [0.1],  # 학습률
    'XGB__max_depth': [9],  # 트리의 최대 깊이
    'XGB__min_child_weight': [1],  # 자식에 필요한 모든 관측치에 대한 가중치 합의 최소
    'XGB__gamma': [1],  # 리프 노드를 추가적으로 나눌지 결정하기 위한 최소 손실 감소
    'XGB__subsample': [0.8],  # 각 트리마다의 관측 데이터 샘플링 비율
    'XGB__colsample_bytree': [0.8],  # 각 트리 구성에 필요한 컬럼(특성) 샘플링 비율
    'XGB__objective': ['multi:softmax'],  # 학습 태스크 파라미터
    'XGB__num_class': [10],  # 분류해야 할 전체 클래스 수, 멀티클래스 분류인 경우 설정
    'XGB__verbosity' : [0]
}


pipe = Pipeline([('MM', MinMaxScaler()),
                ('XGB', XGBClassifier(random_state=3, n_jobs = -1, tree_method = 'gpu_hist', predictor = 'gpu_predictor',  gpu_id=0))])

model = HalvingRandomSearchCV(pipe, parameters,
                    cv = kfold,
                    verbose=1,
                    refit=True,
                    n_jobs=-1,   
                    # n_iter=10 # 디폴트 10
                    factor=2,
                    min_resources=70
                    )

model.fit(X_train, y_train)

end = time.time()
print("최적의 매개변수:",model.best_estimator_)
print("최적의 파라미터:",model.best_params_)
print("best_score:",model.best_score_) 
print("model.score:", model.score(X_test,y_test)) 
