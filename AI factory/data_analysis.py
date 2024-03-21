import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings("ignore")
import glob
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import tensorflow as tf
import keras
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.python.keras import backend as K
import sys
import pandas as pd
from tqdm import tqdm
from keras.preprocessing.image import ImageDataGenerator
import threading
import random
import rasterio
import os
import numpy as np
import sys
from sklearn.utils import shuffle as shuffle_lists
from keras.models import *
from keras.layers import *
import numpy as np
from keras import backend as K
from sklearn.model_selection import train_test_split
import joblib
tf.random.set_seed(3)
np.random.seed(3)

MAX_PIXEL_VALUE = 65535 # 이미지 정규화를 위한 픽셀 최대값

class threadsafe_iter:
    """
    데이터 불러올떼, 호출 직렬화
    """
    def __init__(self, it):
        self.it = it
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def __next__(self):
        with self.lock:
            return self.it.__next__()


def threadsafe_generator(f):
    def g(*a, **kw):
        return threadsafe_iter(f(*a, **kw))

    return g

def get_img_arr(path):
    img = rasterio.open(path).read().transpose((1, 2, 0))    
    img = np.float32(img)/MAX_PIXEL_VALUE
    
    return img

def get_img_762bands(path):
    img = rasterio.open(path).read((7,6,2)).transpose((1, 2, 0))    
    img = np.float32(img)/MAX_PIXEL_VALUE
    
    return img
    
def get_mask_arr(path):
    img = rasterio.open(path).read().transpose((1, 2, 0))
    seg = np.float32(img)
    return seg

images_path = 'C:\\_data\\AI factory\\train_img\\'
masks_path = 'C:\\_data\\AI factory\\train_mask\\' 


@threadsafe_generator
def generator_from_lists(images_path, masks_path, batch_size=32, shuffle = True, random_state=None, image_mode='10bands'):
   
    images = []
    masks = []

    fopen_image = get_img_arr
    fopen_mask = get_mask_arr

    if image_mode == '762':
        fopen_image = get_img_762bands

    i = 0 
    # 데이터 shuffle
    while True:
        
        if shuffle:
            if random_state is None:
                images_path, masks_path = shuffle_lists(images_path, masks_path)
            else:
                images_path, masks_path = shuffle_lists(images_path, masks_path, random_state= random_state + i)
                i += 1 


        for img_path, mask_path in zip(images_path, masks_path):

            img = fopen_image(img_path)
            mask = fopen_mask(mask_path)
            images.append(img)
            masks.append(mask)

            # 이미지 시각화 코드 추가
        if len(images) == 1:  # 예시로 첫 번째 이미지만 시각화합니다.
            plt.figure(figsize=(12, 6))
            plt.subplot(1, 2, 1)
            plt.imshow(img)
            plt.title('Image')
            plt.axis('off')

            plt.subplot(1, 2, 2)
            plt.imshow(mask)
            plt.title('Mask')
            plt.axis('off')

            plt.show()

        if len(images) >= batch_size:
            yield (np.array(images), np.array(masks))
            images = []
            masks = []
def visualize_image_and_mask(image_path, mask_path):
    """
    단일 이미지와 마스크를 로드하고 시각화합니다.
    """
    img = get_img_arr(image_path)
    mask = get_mask_arr(mask_path)

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.title('Image')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(mask, cmap='gray')  # 마스크는 흑백으로 표시
    plt.title('Mask')
    plt.axis('off')

    plt.show()

# 예제 이미지 및 마스크 경로
image_path_example = os.path.join(images_path, 'train_img_0.tif')
mask_path_example = os.path.join(masks_path, 'train_mask_0.tif')

# 시각화 함수 호출
visualize_image_and_mask(image_path_example, mask_path_example)            
            
            
           