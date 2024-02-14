import numpy as np
from sklearn.datasets import load_breast_cancer
from keras.models import Sequential
from keras.layers import Dense
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error,mean_squared_log_error,accuracy_score
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, StandardScaler, RobustScaler
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier


#1. 데이터
datasets= load_breast_cancer()

X = datasets.data
y = datasets.target

columns = datasets.feature_names
X = pd.DataFrame(X,columns=columns)

# print(datasets.feature_names)
# ['mean radius' 'mean texture' 'mean perimeter' 'mean area'
#  'mean smoothness' 'mean compactness' 'mean concavity'
#  'mean concave points' 'mean symmetry' 'mean fractal dimension'
#  'radius error' 'texture error' 'perimeter error' 'area error'
#  'smoothness error' 'compactness error' 'concavity error'
#  'concave points error' 'symmetry error' 'fractal dimension error'
#  'worst radius' 'worst texture' 'worst perimeter' 'worst area'
#  'worst smoothness' 'worst compactness' 'worst concavity'
#  'worst concave points' 'worst symmetry' 'worst fractal dimension']

# RandomForestClassifier : [0.04591717 0.01677276 0.04519297 0.03166873 0.00933174 0.00899141
#  0.03902775 0.11763835 0.00259388 0.00397243 0.00785815 0.00487554
#  0.01326458 0.01993654 0.00575058 0.00491898 0.00545244 0.00329444
#  0.00440461 0.00510406 0.13853722 0.01488148 0.11591062 0.13169502
#  0.0149445  0.02117601 0.04719356 0.10673662 0.00733559 0.00562228]

# X = np.delete(X, 0, axis=1)
# columns = 

# X = pd.DataFrame(X)

# X = X.drop(['mean radius','mean texture'], axis=1)

fi_str = "0.04591717 0.01677276 0.04519297 0.03166873 0.00933174 0.00899141\
 0.03902775 0.11763835 0.00259388 0.00397243 0.00785815 0.00487554\
 0.01326458 0.01993654 0.00575058 0.00491898 0.00545244 0.00329444\
 0.00440461 0.00510406 0.13853722 0.01488148 0.11591062 0.13169502\
 0.0149445  0.02117601 0.04719356 0.10673662 0.00733559 0.00562228"


fi_str = fi_str.split()
fi_float = [float(s) for s in fi_str]
print(fi_float)
fi_list = pd.Series(fi_float)

low_idx_list = fi_list[fi_list <= fi_list.quantile(0.25)].index
print('low_idx_list',low_idx_list)

low_col_list = [X.columns[index] for index in low_idx_list]
if len(low_col_list) > len(X.columns) * 0.25:
    low_col_list = low_col_list[:int(len(X.columns)*0.25)]
print('low_col_list',low_col_list)
X.drop(low_col_list,axis=1,inplace=True)
print("after X.shape",X.shape)






X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, shuffle=False, random_state=3)

mms = MinMaxScaler()
mms.fit(X_train)
X_train = mms.transform(X_train)
X_test = mms.transform(X_test)

models = [
# DecisionTreeClassifier(),
RandomForestClassifier(),
# GradientBoostingClassifier(),
# XGBClassifier()
]


for model in models:
    model_name = model.__class__.__name__
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f"{model_name} accuracy: {accuracy:.4f}")
    print(model.__class__.__name__, ":", model.feature_importances_)




# MinMaxScaler
# 정확도 :  0.9956140350877193
# 로스 :  [0.016668006777763367, 0.9956140518188477]
# R2 :  0.9760378349973726

# MaxAbsScaler
# 정확도 :  0.956140350877193
# 로스 :  [0.5653559565544128, 0.9561403393745422]
# R2 :  0.7603783499737258

# StandardScaler
# 정확도 :  0.9780701754385965
# 로스 :  [0.303570419549942, 0.9780701994895935]
# R2 :  0.8801891749868629

# # RobustScaler
# 정확도 :  0.9692982456140351
# 로스 :  [0.09083357453346252, 0.969298243522644]
# R2 :  0.832264844981608

# linearsvc
# 정확도 :  0.9649122807017544
# model.score :  0.9649122807017544
# R2 :  0.8083026799789806


# RandomForestClassifier accuracy: 0.9605
# RandomForestClassifier : [0.04591717 0.01677276 0.04519297 0.03166873 0.00933174 0.00899141
#  0.03902775 0.11763835 0.00259388 0.00397243 0.00785815 0.00487554
#  0.01326458 0.01993654 0.00575058 0.00491898 0.00545244 0.00329444
#  0.00440461 0.00510406 0.13853722 0.01488148 0.11591062 0.13169502
#  0.0149445  0.02117601 0.04719356 0.10673662 0.00733559 0.00562228]


# RandomForestClassifier accuracy: 0.9561
# RandomForestClassifier : [0.04499761 0.01430554 0.03756057 0.06314609 0.00761784 0.0062954
#  0.03752652 0.09584059 0.00701492 0.01050443 0.01233306 0.00454758
#  0.00567761 0.12652112 0.01843071 0.15467494 0.08769601 0.01851269
#  0.02111569 0.07946392 0.12751869 0.01241231 0.00628616]
