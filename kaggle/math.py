import pandas as pd



path = "c:\\_data\\kaggle\\ai-mathematical-olympiad-prize\\"

train_csv = pd.read_csv(path + "train.csv", index_col=0)
test_csv = pd.read_csv(path + "test.csv", index_col=0)
submission_csv = pd.read_csv(path + "sample_submission.csv")



# print(train_csv)
# print(test_csv)
# print(submission_csv)

# print(train_csv.info())




