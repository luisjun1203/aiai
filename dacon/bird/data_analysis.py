import pandas as pd
import numpy as np
from keras.preprocessing.image import ImageDataGenerator       ###### 이미지를 숫자로 바꿔준다##########
from PIL import Image
import os


path = "c:\\_data\\dacon\\bird\\open\\"

train_csv = pd.read_csv(path + "train.csv", index_col=0)
test_csv = pd.read_csv(path + "test.csv", index_col=0)
submission_csv = pd.read_csv(path + "sample_submission.csv")



print(train_csv)
print(test_csv)
print(submission_csv)



img_path = "c:\\_data\\dacon\\bird\\open\\train\\"
def load_image_example(img_path):
    img = Image.open(img_path)
    img = img.resize((64, 64))
    return img

# 첫 번째 훈련 이미지 경로
first_img_path = train_csv['img_path'].iloc[0]


first_label = train_csv['label'].iloc[0]

print(first_img_path, first_label)











