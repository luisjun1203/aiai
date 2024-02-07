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




