import os
import rasterio
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt


def get_img_arr(path):

    MAX_PIXEL_VALUE = 65535 # 이미지 정규화를 위한 픽셀 최대값

    img = rasterio.open(path).read()#.transpose((1, 2, 0))

    img = np.float32(img)/MAX_PIXEL_VALUE

    for i in range(len(img)):

        max_val = np.max(img[i])

        img[i] = (img[i]/max_val)*255

    return img.astype(np.uint8)

    #return img

def show_images(images, titles=None):

    """

    이미지를 여러 장 보여주는 함수.
    :param images: PIL 이미지 객체의 리스트
    :param titles: 각 이미지에 대한 제목의 리스트 (선택 사항)
    """

    num_images = len(images)

    len_batch = len(images[0])

    print(num_images, len_batch)

    if titles is not None and len(titles) != num_images:

        raise ValueError("이미지와 제목의 수가 일치하지 않습니다.")

   
    fig, axes = plt.subplots(num_images, len_batch, figsize=(50, 50))

    if num_images == 1:

        for j in range(len_batch):

            axes[j].imshow(images[j])

            axes[j].axis('off')

            if titles is not None:

                axes[j].set_title(titles)

    else:

        for i in range(num_images):

            for j in range(len_batch):

                axes[i][j].imshow(images[i][j])

                axes[i][j].axis('off')

                if titles is not None:

                    axes[i][j].set_title(titles[i])

    plt.show()


 

image_path = 'C:\\_data\\AI factory\\train_img\\'
mask_path = 'C:\\_data\\AI factory\\train_mask\\'

images = []


 

for i in range(10):

    image = get_img_arr(os.path.join(image_path, 'train_img_{}.tif'.format(i)))

    mask = get_img_arr(os.path.join(mask_path, 'train_mask_{}.tif'.format(i)))

    images.append(np.concatenate((image, mask), axis=0))

    #break

show_images(images)

print(image.shape)

#img = Image.fromarray(mask)

#img.show()