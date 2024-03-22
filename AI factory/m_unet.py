from keras.applications.vgg19 import VGG19
from keras.models import Model
# from keras.layers import Input
from keras.layers import Layer,Concatenate
from keras.layers import Add
from keras import backend as K
from keras.engine import *
# from keras.utils import multi_gpu_model
from keras.layers import AveragePooling2D
import numpy as np
from keras.models import Model
# from keras.models import Input
from keras.layers import Conv2D, Conv2DTranspose, LeakyReLU, Activation, Concatenate,  Dropout, BatchNormalization, LeakyReLU , Input
from keras.initializers import RandomNormal

from keras.layers import Input, Convolution2D, BatchNormalization, Activation, LeakyReLU, Add, ReLU, GlobalAveragePooling2D, Reshape, AveragePooling2D, UpSampling2D, Flatten
from keras.models import Model
import tensorflow as tf
from keras_self_attention import SeqSelfAttention
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
import os
import numpy as np
import sys
from sklearn.utils import shuffle as shuffle_lists
from keras.models import Model
from keras.layers import Input, Conv2D, BatchNormalization, Activation, UpSampling2D, Concatenate, DepthwiseConv2D
from keras.applications import MobileNetV2
from keras.models import *
from keras.layers import *
import numpy as np
from keras import backend as K
from sklearn.model_selection import train_test_split
import joblib
tf.random.set_seed(3)
np.random.seed(3)
MAX_PIXEL_VALUE = 65535 # 이미지 정규화를 위한 픽셀 최대값
THESHOLDS = 0.25

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
    img = rasterio.open(path).read((7,6,5)).transpose((1, 2, 0))    
    img = np.float32(img)/MAX_PIXEL_VALUE
    
    return img



    
def get_mask_arr(path):
    img = rasterio.open(path).read().transpose((1, 2, 0))
    seg = np.float32(img)
    return seg

#miou metric
def miou(y_true, y_pred, smooth=1e-6):
    # 임계치 기준으로 이진화
    y_pred = tf.cast(y_pred > THESHOLDS, tf.float32)
    
    intersection = tf.reduce_sum(y_true * y_pred, axis=[1, 2, 3])
    union = tf.reduce_sum(y_true, axis=[1, 2, 3]) + tf.reduce_sum(y_pred, axis=[1, 2, 3]) - intersection
    
    # mIoU 계산
    iou = (intersection + smooth) / (union + smooth)
    miou = tf.reduce_mean(iou)
    return miou

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




def define_normal_encoder_block(layer_in, n_filters, batchnorm=True):
	# weight initialization
	init = RandomNormal(stddev=0.02)
	# add downsampling layer
	g = Conv2D(n_filters, (4,4), strides=(1,1), padding='same', kernel_initializer=init)(layer_in)
	# conditionally add batch normalization
	if batchnorm:
		g = BatchNormalization()(g, training=True)
	# leaky relu activation
	g = LeakyReLU(alpha=0.2)(g)
	return g

# define a decoder block
def normal_decoder_block(layer_in, skip_in, n_filters, dropout=True):
	# weight initialization
	init = RandomNormal(stddev=0.02)
	# add upsampling layer
	g = Conv2D(n_filters, (4,4), strides=(1,1), padding='same', kernel_initializer=init)(layer_in)
	# add batch normalization
	g = BatchNormalization()(g, training=True)
	# conditionally add dropout
	if dropout:
		g = Dropout(0.5)(g, training=True)
	# merge with skip connection
	g = Concatenate()([g, skip_in])
	# relu activation
	g = Activation('swish')(g)
	return g


def define_encoder_block(layer_in, n_filters, batchnorm=True):
    init = RandomNormal(stddev=0.02)
    g = Conv2D(n_filters, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(layer_in)
    if batchnorm:
        g = BatchNormalization()(g, training=True)
    g = LeakyReLU(alpha=0.2)(g)
    return g

def decoder_block(layer_in, n_filters, dropout=True):
    init = RandomNormal(stddev=0.02)
    g = Conv2DTranspose(n_filters, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(layer_in)
    g = BatchNormalization()(g, training=True)
    if dropout:
        g = Dropout(0.5)(g, training=True)
    g = Activation('swish')(g)
    return g

def define_generator(image_shape=(256,256,3)):
    init = RandomNormal(stddev=0.02)
    in_image = Input(shape=image_shape)
    
    # Encoder model
    e1 = define_encoder_block(in_image, 64, batchnorm=False)
    e2 = define_encoder_block(e1, 128)
    e3 = define_encoder_block(e2, 256)
    e4 = define_encoder_block(e3, 512)
    e5 = define_encoder_block(e4, 512)
    e6 = define_encoder_block(e5, 512)
    e7 = define_encoder_block(e6, 512)
    
    # Decoder model
    d1 = decoder_block(e7, 512)
    d2 = decoder_block(d1, 512)
    d3 = decoder_block(d2, 512)
    d4 = decoder_block(d3, 512, dropout=False)
    d5 = decoder_block(d4, 256, dropout=False)
    d6 = decoder_block(d5, 128, dropout=False)
    d7 = decoder_block(d6, 64, dropout=False)
    
    # Output
    g = Conv2D(1, (4,4), strides=(1,1), padding='same', kernel_initializer=init)(d7)
    out_image = Activation('sigmoid')(g)
    
    # Define model
    model = Model(in_image, out_image)
    return model
g_model = define_generator(image_shape=(256,256,3))

# Summarize the model
g_model.summary()

def get_model(model_name, input_height=256, input_width=256, n_channels=3, n_filters = 16):
    
    image_shape = (input_height, input_width, n_channels)
    
    # 'jun' 모델을 선택한 경우
    if model_name == 'jun':
        model_func = define_generator
        model = model_func(image_shape=image_shape)
        return model
    
    
model = get_model('jun', input_height=256, input_width=256, n_channels=3)
model.summary()    
    
    
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
EPOCHS = 1000 # 훈련 epoch 지정
BATCH_SIZE = 16  # batch size 지정
IMAGE_SIZE = (256, 256) # 이미지 크기 지정
MODEL_NAME = 'jun' # 모델 이름
RANDOM_STATE = 3 # seed 고정
INITIAL_EPOCH = 0 # 초기 epoch
THESHOLDS = 0.25
lr = 0.001

rlr = ReduceLROnPlateau(monitor='val_miou', patience=5, mode='accuracy', verbose=1, factor=0.5)


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
WORKERS = 22

# 조기종료
EARLY_STOP_PATIENCE = 50

# 중간 가중치 저장 이름
CHECKPOINT_PERIOD = 5
CHECKPOINT_MODEL_NAME = 'checkpoint-{}-{}-epoch_{{epoch:02d}}_03_22_01.hdf5'.format(MODEL_NAME, save_name)
 
# 최종 가중치 저장 이름
FINAL_WEIGHTS_OUTPUT = 'model_{}_{}_final_weights_03_22_01.h5'.format(MODEL_NAME, save_name)

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
loss = sm.losses.binary_focal_dice_loss
# model 불러오기
model = get_model(MODEL_NAME, input_height=IMAGE_SIZE[0], input_width=IMAGE_SIZE[1], n_filters=N_FILTERS, n_channels=N_CHANNELS)
model.compile(optimizer = Adam(learning_rate=lr), loss = loss, metrics = ['accuracy', miou])
model.summary()


# checkpoint 및 조기종료 설정
es = EarlyStopping(monitor='val_miou', mode='max', verbose=1, patience=EARLY_STOP_PATIENCE, restore_best_weights=True)
checkpoint = ModelCheckpoint(os.path.join(OUTPUT_DIR, CHECKPOINT_MODEL_NAME), monitor='val_miou', verbose=1,
save_best_only=True, mode='max', period=CHECKPOINT_PERIOD)

print('---model 훈련 시작---')
history = model.fit_generator(
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
# model.compile(optimizer = Adam(), loss = 'binary_crossentropy', metrics = ['accuracy', miou])
# model.summary()

model.load_weights('C:\\_data\\AI factory\\train_output\\checkpoint-pretrained_attention_unet-base_line-epoch_10_03_22_01.hdf5')


y_pred_dict = {}

for i in test_meta['test_img']:
    img = get_img_762bands(f'C:\\_data\\AI factory\\test_img\\{i}')
    y_pred = model.predict(np.array([img]), batch_size=1, verbose=1)
    
    y_pred = np.where(y_pred[0, :, :, 0] > 0.25, 1, 0) # 임계값 처리
    y_pred = y_pred.astype(np.uint8)
    y_pred_dict[i] = y_pred

joblib.dump(y_pred_dict, 'C:\\_data\\AI factory\\train_output\\y_pred_03_22_01.pkl')    
# def build_discriminator(shape=256,
#                         ndf=64, n_layers=3,
#                         kernel_size=4, strides=2, activation='linear',
#                         n_downsampling=1):
    
#     # img_a_input_512 = Input(shape=(512,512,1))
#     # img_b_input_512 = Input(shape=(512,512,3))

#     img_a_input_256 = Input(shape=(256,256,1))
#     img_b_input_256 = Input(shape=(256,256,3))


#     img_a_input_128 = Input(shape=(128,128,1))
#     img_b_input_128 = Input(shape=(128,128,3))

#     img_a_input_64 = Input(shape=(64,64,1))
#     img_b_input_64 = Input(shape=(64,64,3))
    
#     # if shape==512:
#     #     input_a=img_a_input_512
#     #     input_b=img_b_input_512
        
    
#     if shape==256:
#         input_a=img_a_input_256
#         input_b=img_b_input_256
        
    
#     if shape==128:
#         input_a=img_a_input_128
#         input_b=img_b_input_128
        
    
#     if shape==64:
#         input_a=img_a_input_64
#         input_b=img_b_input_64

#     features = []
#     x = Concatenate(axis=-1)([input_a, input_b])
#     for i in range(n_downsampling):
#         x = AveragePooling2D(3, strides=2, padding='same')(x)

#     x = Conv2D(ndf, kernel_size=kernel_size, strides=2, padding='same')(x)
#     x = LeakyReLU(alpha=0.2)(x)
#     features.append(x)

#     nf = ndf
#     for i in range(1, n_layers):
#         nf = min(ndf * 2, 512)
#         x = Conv2D(nf, kernel_size=kernel_size, strides=2, padding='same')(x)
#         x = BatchNormalization()(x)
#         x = LeakyReLU(alpha=0.2)(x)
#         features.append(x)

#     nf = min(nf * 2, 512)
#     x = Conv2D(nf, kernel_size=kernel_size, strides=1, padding='same')(x)
#     x = BatchNormalization()(x)
#     x = LeakyReLU(alpha=0.2)(x)
#     features.append(x)

#     x = Conv2D(1, kernel_size=kernel_size, strides=1, padding='same')(x)
#     x = Activation(activation)(x)

#     # create model graph
#     model = Model(inputs=[img_a_input_256,img_a_input_128,img_a_input_64, input_b], outputs=[x] + features)
#     print("\nDiscriminator")
#     model.summary()
#     return model

# # model = build_discriminator()
# # model.summary()