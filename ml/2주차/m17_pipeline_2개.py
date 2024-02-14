from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA       # 차원 축소, 주성분 분석


datasets = load_iris()
X = datasets.data
y = datasets.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y )

# scaler = MinMaxScaler()
# X_train = scaler.fit_transform(X_train)
# X_test = scaler.transform(X_test)
# model = RandomForestClassifier()

model = make_pipeline(MinMaxScaler(), StandardScaler(), PCA(), RandomForestClassifier(min_samples_split=2, min_samples_leaf=10, random_state=42))



model.fit(X_train,y_train)

results = model.score(X_test, y_test)

print("model : ", model, ", ",'score : ', results)
y_predict = model.predict(X_test)

acc = accuracy_score(y_test, y_predict)
print("model : ", model, ", ","acc : ", acc)
print("\n")


# model :  Pipeline(steps=[('minmaxscaler', MinMaxScaler()),
#                 ('standardscaler', StandardScaler()), ('pca', PCA()),
#                 ('randomforestclassifier',
#                  RandomForestClassifier(min_samples_leaf=10, random_state=42))]) ,  score :  0.9333333333333333
# model :  Pipeline(steps=[('minmaxscaler', MinMaxScaler()),
#                 ('standardscaler', StandardScaler()), ('pca', PCA()),
#                 ('randomforestclassifier',
#                  RandomForestClassifier(min_samples_leaf=10, random_state=42))]) ,  acc :  0.9333333333333333



