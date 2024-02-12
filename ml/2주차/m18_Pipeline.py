from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.pipeline import make_pipeline, Pipeline
import numpy as np
from sklearn.metrics import accuracy_score
import random
random.seed(42)
np.random.seed(42)

#1 . 데이터
X, y = load_iris(return_X_y=True)
print(X.shape, y.shape)
       
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=123, train_size=0.8, stratify=y)

# print(np.mean(X_train), np.max(X_train))    #3.454583333333333 7.9
# print(np.mean(X_test), np.max(X_test))      #3.504166666666667 7.7
# scaler = MinMaxScaler()
# X_train = scaler.fit_transform(X_train)
# X_test = scaler.transform(X_test)

#2 모델
# model = RandomForestClassifier()
model = make_pipeline(MinMaxScaler(), RandomForestClassifier(min_samples_split=2))
model = Pipeline([("MinMax", MinMaxScaler())
                  , ("RF", RandomForestClassifier())])
  
#3 훈련
model.fit(X_train, y_train)

#4 평가
results = model.score(X_test, y_test)
print('model : ', " acc :", results)
y_predict = model.predict(X_test)

acc = accuracy_score(y_test, y_predict)
print("model : ", model, ", ","acc : ", acc)

# model :   acc : 0.9666666666666667

# model :   acc : 0.9666666666666667
# model :  Pipeline(steps=[('MinMax', MinMaxScaler()), ('RF', RandomForestClassifier())]) ,  acc :  0.9666666666666667