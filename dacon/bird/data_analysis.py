import pandas as pd
import numpy as np
from keras.preprocessing.image import ImageDataGenerator       ###### 이미지를 숫자로 바꿔준다##########



path = "c:\\_data\\dacon\\bird\\open\\"

train_csv = pd.read_csv(path + "train.csv", index_col=0)
test_csv = pd.read_csv(path + "test.csv", index_col=0)
submission_csv = pd.read_csv(path + "sample_submission.csv")



print(train_csv)
print(test_csv)
print(submission_csv)

from PIL import Image
import os

img_path = "c:\\_data\\dacon\\bird\\open\\train\\"
# 이미지 불러오기 예제 함수 정의
def load_image_example(img_path):
    # 이미지 불러오기
    img = Image.open(img_path)
    # 이미지 크기 조정 (예: 224x224)
    img = img.resize((64, 64))
    return img

# 첫 번째 훈련 이미지 경로
first_img_path = train_csv['img_path'].iloc[0]

# 이미지가 실제로 해당 경로에 없으므로 예제 코드를 실행할 수는 없습니다.
# 이미지 불러오기 예제를 위한 코드 (실행 불가)
# first_img = load_image_example(first_img_path)

# 첫 번째 이미지에 해당하는 레이블
first_label = train_csv['label'].iloc[0]

print(first_img_path, first_label)











