import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from catboost import CatBoostClassifier

import warnings

warnings.filterwarnings('ignore')

# 1. 데이터
X, y = load_breast_cancer(return_X_y=True)


X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=777, train_size=0.8, stratify=y)


scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 2. 모델
xgb = XGBClassifier()
rf = RandomForestClassifier()
lr = LogisticRegression()
# rf2 = RandomForestClassifier()

models = [xgb, rf, lr]
li = []
li2 = []

for model in models:
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    # print(y_pred.shape)         #(455,)
    
    li.append(y_pred)
    li2.append(y_pred_test)
    
    score = accuracy_score(y_test, y_pred_test)
    class_name = model.__class__.__name__
    print("{0} ACC : {1:.4f}".format(class_name, score))
    
###################### 민형이 방식 ####################################################
'''
X_train_stack = []
X_test_stack = []    

for model in models:
    X_train_stack.append(model.predict(X_train))    
    X_test_stack.append(model.predict(X_test))
    
# x_train_stack = np.array(x_train_stack)
# x_test_stack = np.array(x_test_stack)

# print(x_train_stack.shape, x_test_stack.shape)   # (3, 455) (3, 114)
X_train_stack = np.array(X_train_stack).T
X_test_stack = np.array(X_test_stack).T
# print(x_train_stack.shape, x_test_stack.shape)   # (455, 3) (114, 3)

model2 = CatBoostClassifier(verbose=0)
model2.fit(X_train_stack,y_train)
y_pred2 = model2.predict(X_test_stack)
score2 = accuracy_score(y_test, y_pred2)
print("스태킹 결과 :", score2)    
    
# XGBClassifier ACC : 0.9912
# RandomForestClassifier ACC : 0.9825
# LogisticRegression ACC : 0.9737
# 스태킹 결과 : 0.9824561403508771
'''
############################# 영현씨 방식 ###############################################
def self_Stacking(models:list[tuple], final_model, X_train, X_test, y_train, y_test):
    pred_list = []
    trained_model_dict = {}
    for name, model in models:
        model.fit(X_train,y_train)
        pred = model.predict(X_train)       # x_test로 하면 나중에 final_model도 테스트 셋으로 학습해야하기에 테스트 셋 의미가 없어진다
        pred_list.append(pred)              # 예측값 저장 쉽게 append로
        trained_model_dict[name] = model    # 훈련 완료된 모델들 저장, 출력을 위해서 이름도 같이 저장
        
    stacked_train_pred = np.asarray(pred_list).T    # (n,3)형태를 위해서 Transpose
    final_model.fit(stacked_train_pred,y_train)     
    
    pred_list = []
    print_dict = {}
    for name, model in trained_model_dict.items():  # 딕셔러니에서 키값과 내용물을 같이 반환
        pred = model.predict(X_test)   
        result = model.score(X_test,y_test)
        pred_list.append(pred)              # 예측값 저장 쉽게 append로
        print_dict[f'{name} ACC'] = result  # 이름과 함께 ACC 저장
    
    stacked_test_pred = np.asarray(pred_list).T
    final_result = final_model.score(stacked_test_pred,y_test)
    
    for name , acc in print_dict.items():
        print(name,":",acc)
    print("스태킹 결과: ",final_result)
    
self_Stacking([
    ('xgb',XGBClassifier()),
    ('RF',RandomForestClassifier()),
    ('LR',LogisticRegression()),
],CatBoostClassifier(verbose=0),X_train,X_test,y_train,y_test)

#################### 선생님 방식 ##################################
'''
xgb = XGBClassifier()
rf = RandomForestClassifier()
lr = LogisticRegression()

models = [xgb, rf, lr]
li = []
li2 = []

for model in models:
    model.fit(X_train, y_train)
    y_pred = model.predict(X_train)
    y_predict = model.predict(X_test)
    score = accuracy_score(y_test, y_predict)
    li.append(y_pred)
    li2.append(y_predict)
    class_name = model.__class__.__name__
    print("{0} 정확도 : {1:.4f}".format(class_name, score))

new_X_train = np.array(li).T
new_X_test = np.array(li2).T

model2 = CatBoostClassifier(verbose=0)
model2.fit(new_X_train, y_train)
y_pred3 = model2.predict(new_X_test)
score2 = accuracy_score(y_test, y_pred3)
print("스태킹 결과 : ", score2)

# XGBClassifier 정확도 : 0.9912
# RandomForestClassifier 정확도 : 0.9825
# LogisticRegression 정확도 : 0.9737
# 스태킹 결과 :  0.9824561403508771
'''


