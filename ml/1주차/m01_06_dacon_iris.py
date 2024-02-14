# https://dacon.io/competitions/open/235610/mysubmission



import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_absolute_error, accuracy_score
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import OneHotEncoder
from keras.utils import to_categorical
from sklearn.svm import LinearSVC 



path = "c:\\_data\\dacon\\iris\\"

train_csv = pd.read_csv(path + "train.csv", index_col=0)        
                                             
test_csv = pd.read_csv(path + "test.csv", index_col=0)

submission_csv = pd.read_csv(path + "sample_submission.csv")



X = train_csv.drop(['species'], axis=1)
y = train_csv['species']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, shuffle=True, random_state=3, stratify=y)

model = LinearSVC(C=10000, random_state=3, verbose=1)

model.fit(X_train, y_train)


results = model.score(X_test, y_test)
print("model.score : ", results)


y_submit = model.predict(test_csv)  
y_predict = model.predict(X_test) 

# y_test = np.argmax(y_test, axis=1)
# y_submit = np.argmax(y_submit, axis=1)
submission_csv['species'] = y_submit       
                    

submission_csv.to_csv(path + "submission_02_07_1_.csv", index=False)
# print(submission_csv)
acc = accuracy_score(y_test, y_predict)
print("accuracy_score : ", acc)
    
# model.score :  0.9583333333333334
# accuracy_score :  0.9583333333333334











