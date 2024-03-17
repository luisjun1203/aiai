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
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.python.keras import backend as K
import sys
import pandas as pd
from tqdm import tqdm
from keras.preprocessing.image import ImageDataGenerator
import threading
import random
import rasterio
import numpy as np
import sys
import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, Conv2DTranspose
from keras.layers import Activation, BatchNormalization, Dropout, Lambda, Multiply, Add
from keras.models import Model
from keras import backend as K
from sklearn.utils import shuffle as shuffle_lists
from keras.models import *
from keras.layers import *
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

            if len(images) >= batch_size:
                yield (np.array(images), np.array(masks))
                images = []
                masks = []
                





# 이미지 패치 생성 및 인코딩
class Patches(layers.Layer):
    def __init__(self, patch_size):
        super(Patches, self).__init__()
        self.patch_size = patch_size

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding='VALID',
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches

    def get_config(self):
        config = super().get_config()
        config.update({"patch_size": self.patch_size})
        return config

class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, projection_dim):
        super(PatchEncoder, self).__init__()
        self.num_patches = num_patches
        self.projection_dim = projection_dim
        self.projection = layers.Dense(units=projection_dim)
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim)

    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded   
      
    def get_config(self):
        config = super().get_config()
        config.update({
            "num_patches": self.num_patches,
            "projection_dim": self.projection_dim
        })
        return config
    
class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.rate = rate
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = keras.Sequential([
            layers.Dense(ff_dim, activation="relu"), 
            layers.Dense(embed_dim),
        ])
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training=False):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)    

    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "ff_dim": self.ff_dim,
            "rate": self.rate,
        })
        return config    
    

    
def create_vit_segmentator():
    inputs = layers.Input(shape=(256, 256, 3))
    
    # 이미지를 16x16 패치로 나눔
    patches = Patches(patch_size=16)(inputs)  # 패치 크기를 16으로 수정
    # 패치를 인코딩
    num_patches = ((256 // 16) * (256 // 16))  # 16x16 패치로 나누면, 256//16 = 16, 즉 16*16=256 패치가 생성됨
    encoded_patches = PatchEncoder(num_patches=num_patches, projection_dim=768)(patches)  
    
    # Transformer 블록 추가
    for _ in range(4):  
        encoded_patches = TransformerBlock(768, 8, 2048)(encoded_patches)
    
    # 패치를 원본 이미지 크기로 복원
    # Reshape는 이제 16x16 패치에 대한 256 패치를 기반으로 하므로 수정 필요 없음
    x = layers.Reshape((16, 16, 768))(encoded_patches)
    # Conv2DTranspose 레이어들을 조정하여 최종 출력 크기가 256x256이 되도록 함
    # 첫 번째 Conv2DTranspose 레이어는 이미지를 16배로 확대하므로 strides를 (16, 16)으로 설정
    x = layers.Conv2DTranspose(128, 3, strides=(16, 16), padding="same", activation="relu")(x)
    # 두 번째 Conv2DTranspose 레이어는 필요 없으므로 삭제하고, 마지막 레이어만 유지
    x = layers.Conv2DTranspose(1, 3, strides=(1, 1), padding="same", activation="sigmoid")(x)  # 마지막 Conv2DTranspose 조정
    
    model = keras.Model(inputs=inputs, outputs=x)
    return model

model = create_vit_segmentator()
model.summary()               


def get_model(model_name, input_height=256, input_width=256, n_channels=3):
    if model_name == 'VIT':
        # AttentionUNet 모델 구현을 여기에 추가하거나 대체 모델을 사용합니다.
        # 예시로 Vision Transformer 기반 세그멘테이터를 사용합니다.
        model = create_vit_segmentator()
    else:
        raise ValueError("Model name not recognized.")
    return model
    
    # return model(
    #         nClasses      = nClasses,  
    #         input_height  = input_height, 
    #         input_width   = input_width,
    #         n_filters     = n_filters,
    #         dropout       = dropout,
    #         batchnorm     = batchnorm,
    #         n_channels    = n_channels
    #     )
# 두 샘플 간의 유사성 metric
def dice_coef(y_true, y_pred, smooth=1):
    intersection = K.sum(y_true * y_pred, axis=[1,2,3])
    union = K.sum(y_true, axis=[1,2,3]) + K.sum(y_pred, axis=[1,2,3])
    dice = K.mean((2. * intersection + smooth)/(union + smooth), axis=0)
    return dice

# 픽셀 정확도를 계산 metric
def pixel_accuracy (y_true, y_pred):
    sum_n = np.sum(np.logical_and(y_pred, y_true))
    sum_t = np.sum(y_true)
 
    if (sum_t == 0):
        pixel_accuracy = 0
    else:
        pixel_accuracy = sum_n / sum_t
    return pixel_accuracy                   







# 사용할 데이터의 meta정보 가져오기

train_meta = pd.read_csv('C:\\_data\\AI factory\\train_meta.csv')
test_meta = pd.read_csv('C:\\_data\\AI factory\\test_meta.csv')


# 저장 이름
save_name = 'base_line'

N_FILTERS = 16 # 필터수 지정
N_CHANNELS = 3 # channel 지정
EPOCHS = 100 # 훈련 epoch 지정
BATCH_SIZE = 8   # batch size 지정
IMAGE_SIZE = (256, 256) # 이미지 크기 지정
MODEL_NAME = 'VIT' # 모델 이름
RANDOM_STATE = 3 # seed 고정
INITIAL_EPOCH = 0 # 초기 epoch
THESHOLDS = 0.25
lr = 0.1

rlr = ReduceLROnPlateau(monitor='val_miou', patience=10, mode='accuracy', verbose=1, factor=0.5)


def miou(y_true, y_pred, smooth=1e-6):
    # 임계치 기준으로 이진화
    y_pred = tf.cast(y_pred > THESHOLDS, tf.float32)
    
    intersection = tf.reduce_sum(y_true * y_pred, axis=[1, 2, 3])
    union = tf.reduce_sum(y_true, axis=[1, 2, 3]) + tf.reduce_sum(y_pred, axis=[1, 2, 3]) - intersection
    
    # mIoU 계산
    iou = (intersection + smooth) / (union + smooth)
    miou = tf.reduce_mean(iou)
    return miou

# 데이터 위치
IMAGES_PATH = 'C:\\_data\\AI factory\\train_img\\'
MASKS_PATH = 'C:\\_data\\AI factory\\train_mask\\'

# 가중치 저장 위치
OUTPUT_DIR = 'C:\_data\AI factory\\train_output\\'
WORKERS = 20

# 조기종료
EARLY_STOP_PATIENCE = 20

# 중간 가중치 저장 이름
CHECKPOINT_PERIOD = 10
CHECKPOINT_MODEL_NAME = 'checkpoint-{}-{}-epoch_{{epoch:02d}}_03_16_01.hdf5'.format(MODEL_NAME, save_name)
 
# 최종 가중치 저장 이름
FINAL_WEIGHTS_OUTPUT = 'model_{}_{}_final_weights_03_16_01.h5'.format(MODEL_NAME, save_name)

# 사용할 GPU 이름
CUDA_DEVICE = 0


# 저장 폴더 없으면 생성
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)


# GPU 설정
os.environ["CUDA_VISIBLE_DEVICES"] = str(CUDA_DEVICE)
try:
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.compat.v1.Session(config=config)
    K.set_session(sess)
except:
    pass

try:
    np.random.bit_generator = np.random._bit_generator
except:
    pass


# train : val = 8 : 2 나누기
x_tr, x_val = train_test_split(train_meta, test_size=0.1, random_state=RANDOM_STATE)
print(len(x_tr), len(x_val))

# train : val 지정 및 generator
images_train = [os.path.join(IMAGES_PATH, image) for image in x_tr['train_img'] ]
masks_train = [os.path.join(MASKS_PATH, mask) for mask in x_tr['train_mask'] ]

images_validation = [os.path.join(IMAGES_PATH, image) for image in x_val['train_img'] ]
masks_validation = [os.path.join(MASKS_PATH, mask) for mask in x_val['train_mask'] ]

train_generator = generator_from_lists(images_train, masks_train, batch_size=BATCH_SIZE, random_state=RANDOM_STATE, image_mode="762")
validation_generator = generator_from_lists(images_validation, masks_validation, batch_size=BATCH_SIZE, random_state=RANDOM_STATE, image_mode="762")


import segmentation_models as sm
# sm.set_framework('tf.keras')
loss = sm.losses.binary_focal_jaccard_loss
# model 불러오기
model = get_model(MODEL_NAME, input_height=IMAGE_SIZE[0], input_width=IMAGE_SIZE[1], n_channels=N_CHANNELS)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), loss=loss, metrics=['accuracy', miou])
model.summary()


checkpoint = ModelCheckpoint(os.path.join(OUTPUT_DIR, CHECKPOINT_MODEL_NAME), monitor='val_miou', verbose=1, save_best_only=True, mode='max', save_freq='epoch')

es = EarlyStopping(monitor='val_miou', mode='max', verbose=1, patience=EARLY_STOP_PATIENCE, restore_best_weights=True)

rlr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, verbose=1, mode='min')

history = model.fit(
    train_generator,
    steps_per_epoch=len(images_train) // BATCH_SIZE,
    validation_data=validation_generator,
    validation_steps=len(images_validation) // BATCH_SIZE,
    callbacks=[checkpoint, es, rlr],
    epochs=EPOCHS,
    workers=WORKERS,
    initial_epoch=INITIAL_EPOCH,
)

print('---model 훈련 종료---')

print('가중치 저장')
model_weights_output = os.path.join(OUTPUT_DIR, FINAL_WEIGHTS_OUTPUT)
model.save_weights(model_weights_output)
print("저장된 가중치 명: {}".format(model_weights_output))


# model = get_model(MODEL_NAME, input_height=IMAGE_SIZE[0], input_width=IMAGE_SIZE[1], n_filters=N_FILTERS, n_channels=N_CHANNELS)
# model.compile(optimizer = Adam(), loss=loss, metrics = ['accuracy', miou])
# model.summary()

model.load_weights('C:\\_data\\AI factory\\train_output\\model_AttentionUNet_base_line_final_weights_03_16_01.h5')


y_pred_dict = {}

for i in test_meta['test_img']:
    img = get_img_762bands(f'C:\\_data\\AI factory\\test_img\\{i}')
    y_pred = model.predict(np.array([img]), batch_size=1, verbose=1)
    
    y_pred = np.where(y_pred[0, :, :, 0] > 0.25, 1, 0) # 임계값 처리
    y_pred = y_pred.astype(np.uint8)
    y_pred_dict[i] = y_pred

joblib.dump(y_pred_dict, 'C:\\_data\\AI factory\\train_output\\y_pred_03_16_01.pkl')

