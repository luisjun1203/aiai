# keras 18_01 복사

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_absolute_error, accuracy_score
from scipy.special import softmax
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier


class CustomXGBClassfier(XGBClassifier):        # XGBClassfier을 상속받겠다.
    def __str__(self):
        return 'XGBClassifier()'                

aaa = CustomXGBClassfier()
# aaa는 인스턴스

# 1.데이터

X, y = load_iris(return_X_y=True)
print(X.shape, y.shape) #(150, 4) (150,)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=713, stratify=y)

# 2. 모델구성

# model = DecisionTreeClassifier(random_state=666)
# model = RandomForestClassifier(random_state=666)
# model = GradientBoostingClassifier(random_state=666)
# model = XGBClassifier()
models = [
DecisionTreeClassifier(),
RandomForestClassifier(),
GradientBoostingClassifier(),
aaa
]


for model in models:
    model_name = model.__class__.__name__
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f"{model_name} accuracy: {accuracy:.4f}")
    print(model, ":", model.feature_importances_)





# # 3. 컴파일, 훈련
# model.fit(X_train, y_train)

# # 4. 평가, 예측
# results = model.score(X_test, y_test)
# print("model.score : ", results)

# y_predict = model.predict(X_test)
# print(y_predict)

# acc = accuracy_score(y_test, y_predict)
# print(model, "accuracy_score : ", acc)

# print(model, ":", model.feature_importances_)

# DecisionTreeClassifier(random_state=666) accuracy_score :  0.9666666666666667
# DecisionTreeClassifier(random_state=666) : [0.01666667 0.         0.07407537 0.90925796]

# RandomForestClassifier(random_state=666) accuracy_score :  0.9666666666666667
# RandomForestClassifier(random_state=666) : [0.08245144 0.02513798 0.48869634 0.40371424]

# GradientBoostingClassifier(random_state=666) accuracy_score :  0.9666666666666667
# GradientBoostingClassifier(random_state=666) : [0.00822725 0.01484092 0.55791586 0.41901596]



# LinearSVC accuracy: 0.9333
# Perceptron accuracy: 0.6667
# LogisticRegression accuracy: 0.9667
# KNeighborsClassifier accuracy: 0.9667
# DecisionTreeClassifier accuracy: 0.9667
# RandomForestClassifier accuracy: 0.9667






# DecisionTreeClassifier accuracy: 0.9667
# DecisionTreeClassifier : [0.         0.01666667 0.57407537 0.40925796]
# RandomForestClassifier accuracy: 0.9667
# RandomForestClassifier : [0.09643216 0.02975015 0.42475327 0.44906443]
# GradientBoostingClassifier accuracy: 0.9667
# GradientBoostingClassifier : [0.00986647 0.01429541 0.61321158 0.36262654]
# XGBClassifier accuracy: 0.9667
# XGBClassifier : [0.0122154  0.02133699 0.73724246 0.22920507]


