

import matplotlib.pyplot as plt
import rasterio
import numpy as np

MAX_PIXEL_VALUE = 65535

IMAGES_PATH = 'C:\\_data\\AI factory\\train_img\\'
MASKS_PATH = 'C:\\_data\\AI factory\\train_mask\\'

# 밴드 이미지와 마스크 이미지 비교
def show_band_images(image_path, mask_path):
    fig, axs = plt.subplots(3, 4, figsize=(20, 12))
    axs = axs.ravel()
    
    for i in range(10):
        img = rasterio.open(image_path).read(i+1).astype(np.float32) / MAX_PIXEL_VALUE
        axs[i].imshow(img)
        axs[i].set_title(f'Band {i+1}')
        axs[i].axis('off')
    
    img = rasterio.open(mask_path).read(1).astype(np.float32) / MAX_PIXEL_VALUE
    axs[10].imshow(img)
    axs[10].set_title('Mask Image')
    axs[10].axis('off')
    axs[11].axis('off')
    plt.suptitle('Band images compared to Mask image')
    plt.tight_layout()
    plt.show() 

# 밴드 조합 이미지 확인
def show_bands_image(image_path, band=(0, 0, 0)):
    img = rasterio.open(image_path).read(band).transpose((1, 2, 0))
    img = np.float32(img) / MAX_PIXEL_VALUE
    plt.figure(figsize=(10, 10))
    plt.imshow(img)
    plt.axis('off')  # 축 표시 없애기
    plt.title(f'Band {band} combined image')
    plt.show()
    return img

for i in range(1004, 1007):
    # 데이터 확인
    show_band_images(IMAGES_PATH + f'train_img_{i}.tif', MASKS_PATH + f'train_mask_{i}.tif')
    show_bands_image(IMAGES_PATH + f'train_img_{i}.tif', (10, 7, 2))