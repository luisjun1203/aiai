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
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


#1. 데이터
datasets= load_breast_cancer()

X = datasets.data
y = datasets.target

# columns = datasets.feature_names
# X = pd.DataFrame(X,columns=columns)

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

# fi_str = "0.04591717 0.01677276 0.04519297 0.03166873 0.00933174 0.00899141\
#  0.03902775 0.11763835 0.00259388 0.00397243 0.00785815 0.00487554\
#  0.01326458 0.01993654 0.00575058 0.00491898 0.00545244 0.00329444\
#  0.00440461 0.00510406 0.13853722 0.01488148 0.11591062 0.13169502\
#  0.0149445  0.02117601 0.04719356 0.10673662 0.00733559 0.00562228"


# fi_str = fi_str.split()
# fi_float = [float(s) for s in fi_str]
# print(fi_float)
# fi_list = pd.Series(fi_float)

# low_idx_list = fi_list[fi_list <= fi_list.quantile(0.25)].index
# print('low_idx_list',low_idx_list)

# low_col_list = [X.columns[index] for index in low_idx_list]
# if len(low_col_list) > len(X.columns) * 0.25:
#     low_col_list = low_col_list[:int(len(X.columns)*0.25)]
# print('low_col_list',low_col_list)
# X.drop(low_col_list,axis=1,inplace=True)
# print("after X.shape",X.shape)
sts = StandardScaler()
X = sts.fit_transform(X)



n_features = len(np.unique(y))
    # print(n_features)

accuracy_results = {}

for n in range(1, n_features):
    lda = LinearDiscriminantAnalysis(n_components=n)
    X1 = lda.fit_transform(X, y)

    X_train, X_test, y_train, y_test = train_test_split(X1, y, test_size=0.4, shuffle=False, random_state=3)

    


   


    # X_train_p = lda.fit_transform(X_train)
    # X_test_p = lda.transform(X_test)  
    # # y_train_p = lda.fit_transform(y_train)
    # # y_test_p = lda.transform(y_test)    

    model = RandomForestClassifier(random_state=3)
    model.fit(X_train, y_train)
    y_predict = model.predict(X_test)
    acc = accuracy_score(y_test, y_predict)
    
   
    accuracy_results[n] = acc
    print(f"n_components = {n}, Accuracy: {acc}")
    
# n_components = 1, Accuracy: 0.956140350877193
# EVR = lda.explained_variance_ratio_         
# print(EVR)

# evr_cumsum = np.cumsum(EVR)
# print(evr_cumsum)   # 0.9864315

# print(sum(EVR))     


