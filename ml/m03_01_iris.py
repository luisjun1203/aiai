# keras 18_01 복사

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_absolute_error, accuracy_score
from scipy.special import softmax
from sklearn.svm import LinearSVC,SVC
from sklearn.linear_model import Perceptron, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


# 1.데이터

X, y = load_iris(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=713, stratify=y)

# 2. 모델구성

models = [
LinearSVC(),
Perceptron(),
LogisticRegression(),
KNeighborsClassifier(),
DecisionTreeClassifier(),
RandomForestClassifier()
]

for model in models:
    model_name = model.__class__.__name__
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f"{model_name} accuracy: {accuracy:.4f}")



# 3. 컴파일, 훈련
# model.fit(X_train, y_train)

# # 4. 평가, 예측
# results = model.score(X_test, y_test)
# print("model.score : ", results)

# y_predict = model.predict(X_test)
# print(y_predict)

# acc = accuracy_score(y_test, y_predict)
# print("acc : ", acc)



# LinearSVC accuracy: 0.9333
# Perceptron accuracy: 0.6667
# LogisticRegression accuracy: 0.9667
# KNeighborsClassifier accuracy: 0.9667
# DecisionTreeClassifier accuracy: 0.9667
# RandomForestClassifier accuracy: 0.9667









