from sklearn.datasets import load_diabetes
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import time
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder



datasets = load_diabetes()
X = datasets.data
y = datasets.target


columns = datasets.feature_names
X = pd.DataFrame(X,columns=columns)


fi_str = "0.11132886 0.02855667 0.12190132 0.11529632 0.11109842 0.11213265\
 0.10702958 0.05943837 0.12141634 0.11180148"


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


lae = LabelEncoder()
lae.fit(y)
y = lae.transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=1226)           # 1226 713

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
    r2 = r2_score(y_test, predictions)
    print(f"{model_name} accuracy: {r2:.4f}")
    print(model.__class__.__name__, ":", model.feature_importances_)

# loss = 'mse' , random_state=1226, epochs=50, batch_size=10, test_size=0.1
# 로스 :  2031.322509765625+
# R2스코어 :  0.7246106597957658

# model.score :  0.022222222222222223
# R2스코어 :  0.5699312314693923

# RandomForestClassifier accuracy: 0.4862
# RandomForestClassifier : [0.11132886 0.02855667 0.12190132 0.11529632 0.11109842 0.11213265
#  0.10702958 0.05943837 0.12141634 0.11180148]


# RandomForestClassifier accuracy: 0.1617
# RandomForestClassifier : [0.12854363 0.1383427  0.12700509 0.12773477 0.13306956 0.08066242
#  0.13709115 0.12755068]