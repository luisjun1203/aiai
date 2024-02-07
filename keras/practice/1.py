import pandas as pd
import numpy as np
import copy
import re

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split as tts
from xgboost import XGBClassifier

path = "c:\\_data\\dacon\\loan\\"

train = pd.read_csv(path + "train.csv")
test = pd.read_csv(path + "test.csv")
submission = pd.read_csv(path + "sample_submission.csv")
def only_int(string):
  num=re.sub(r'[^0-9]', '', string)
  if num:
    return num
  else:
    return "0"
def extract_categorical_columns(df):
    data = []
    for e, i in enumerate(df.columns):
        if df[i].dtypes == 'object':
            data.append(i)
    return data
def ordinal_encoding(train_df, test_df, categorical_columns):
    from sklearn.preprocessing import OrdinalEncoder
    train, test = train_df.copy(), test_df.copy()
    data = {}
    for col in categorical_columns:
        ordinal_encoder = OrdinalEncoder(handle_unknown='use_encoded_value',
                                         unknown_value=-1)
        ordinal_encoder.fit(train[col].values.reshape(-1, 1))
        train[col] = ordinal_encoder.transform(train[col].values.reshape(-1, 1)).reshape(-1)
        if col in test:
            test[col] = ordinal_encoder.transform(test[col].values.reshape(-1, 1)).reshape(-1)
        data[col] = ordinal_encoder
    return train, test, data

def sep_ml_xy(df, target):
    y = df[target]
    x = df.drop(columns=target)
    return x, y

def ml_train_valid(model, metric, metric_options, train_data, train_target, test_data, test_target):
    model = model.fit(train_data, train_target)
    pred = model.predict(test_data)
    evaluate = metric(test_target, pred, **metric_options)
    return pred, evaluate, model

def ml_predict(model, test_data):
    pred = model.predict(test_data)
    return pred

train = train.drop(columns=['ID'])
test = test.drop(columns=['ID'])
for span in ["대출기간", "근로기간"]:
    train[span] = train[span].apply(lambda x: int(only_int(x)))
    test[span] = test[span].apply(lambda x: int(only_int(x)))

categorical_columns = extract_categorical_columns(train)
train, test, ord_dict = ordinal_encoding(train, test, categorical_columns)
train_x, train_y = sep_ml_xy(train, "대출등급")
train_x, valid_x, train_y, valid_y = tts(train_x, train_y, train_size=0.8, shuffle=True, random_state=3)
model = XGBClassifier()
_, evaluate, model = ml_train_valid(model, f1_score, {"average": "macro"},
                                      train_x, train_y, valid_x, valid_y)
print("valid score:", evaluate)
pred = ml_predict(model, test)
submission['대출등급'] = ord_dict["대출등급"].inverse_transform(pred.reshape(-1, 1)).reshape(-1)
print(submission)

submission.to_csv(path + "submission_0119_00_.csv", index=False)