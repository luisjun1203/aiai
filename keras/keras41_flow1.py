import sys
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.utils.image_utils import img_to_array, load_img
import numpy as np


print("텐서플로 버전 : ", tf.__version__)   # 텐서플로 버전 :  2.9.0
print("파이썬 버전 : ", sys.version)        # 파이썬 버전 :  3.9.18 

from keras.preprocessing.image import ImageDataGenerator
# from tensorflow.keras.preprocessing.image import load_img         # 이미지 땡겨온다
# from tensorflow.keras.preprocessing.image import img_to_array     # 이미지를 수치화


path = "c:\\_data\\image\\cat_and_dog\\train\\Cat\\1.jpg"
img = load_img(path, 
                 target_size=(150, 150),
                 
                 )
print(img)
# <PIL.Image.Image image mode=RGB size=150x150 at 0x20E81F98A90>
print(type(img))    # <class 'PIL.Image.Image'>
# plt.imshow(img)
# plt.show()

arr = img_to_array(img)
print(arr)
print(arr.shape)    # (150, 150, 3)     # 원본 사이즈 : (281, 300, 3)
print(type(arr))    # <class 'numpy.ndarray'>

# 차원 증가
img = np.expand_dims(arr, axis=0)          # 축에 따라 들어감
print(img.shape)                # (1, 150, 150, 3)


#################### 여기부터 증폭 ############################################
datagen = ImageDataGenerator(
    # rescale=1./255,
    horizontal_flip=True,
    vertical_flip=True,
    
)

it = datagen.flow(img,               
                  batch_size=1,
                  
                  )

fig, ax = plt.subplots(nrows=2, ncols=5, figsize=(10,10))           # subplot : 여러장을 한번에 볼수있음 (2행 5열)
# matplotlib 사용하여 이미지 표시할 subplot 생성
# fig = figures : 데이터가 담기는 프레임
# ax = axes : 실제 데이터가 그려지는 캔버스 ,모든 plot은 axes 위에서 이루어져야 한다

for i in range(2):
     for j in range(5):
        batch = it.next()                   # 증강된 데이터 생성해서 batch에 저장
        image = batch[0].astype('uint8')       # (150,150,3)        # batch 첫번째 이미지 나타냄
    
        ax[i, j].imshow(image)     # 이미지가 ax라는 놈에 들어간다 i번째 서브플롯에 이미지 표시
        ax[i, j].axis('off')       # 해당 subplot 축을 숨김
print(np.min(batch), np.max(batch))    
plt.show()    








