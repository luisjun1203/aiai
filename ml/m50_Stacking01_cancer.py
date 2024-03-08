import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, BaggingClassifier, StackingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# data
X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X,y,train_size=0.9,random_state=47)

sclaer = MinMaxScaler().fit(X_train)
x_train = sclaer.transform(X_train)
x_test = sclaer.transform(X_test)

model = StackingClassifier([
    ('xgb',XGBClassifier()),
    ('RF',RandomForestClassifier()),
    ('LG',LogisticRegression()),
],final_estimator=CatBoostClassifier())

model.fit(X_train,y_train)
result = model.score(X_test,y_test)
print("ACC : ",result)

# ACC :  0.9649122807017544