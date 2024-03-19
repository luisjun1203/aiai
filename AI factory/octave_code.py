
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
from keras import layers
from keras.models import Model
import keras.backend as K
from keras.utils import conv_utils
from keras.layers import Conv2D, AvgPool2D, Dropout, Input, BatchNormalization, AveragePooling2D, Add
from keras.layers import Cropping2D, UpSampling2D, Conv2DTranspose, concatenate, Concatenate, Lambda
from keras.models import Model
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
import keras.backend as K
from sklearn.model_selection import train_test_split
from keras.optimizers import SGD, Adam
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
    img = rasterio.open(path).read((7,6,2)).transpose((1, 2, 0))    
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
                

num_classes = 1

class OctConv2D(layers.Layer):
    def __init__(self, filters, alpha, kernel_size=(3,3), strides=(1,1), 
                    padding="same", kernel_initializer='glorot_uniform',
                    kernel_regularizer=None, kernel_constraint=None,
                    **kwargs):
        """
        OctConv2D : Octave Convolution for image( rank 4 tensors)
        filters: # output channels for low + high
        alpha: Low channel ratio (alpha=0 -> High only, alpha=1 -> Low only)
        kernel_size : 3x3 by default, padding : same by default
        """
        assert alpha >= 0 and alpha <= 1
        assert filters > 0 and isinstance(filters, int)
        super().__init__(**kwargs)

        self.alpha = alpha
        self.filters = filters
        # optional values
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.kernel_initializer = kernel_initializer
        self.kernel_regularizer = kernel_regularizer
        self.kernel_constraint = kernel_constraint
        # -> Low Channels 
        self.low_channels = int(self.filters * self.alpha)
        # -> High Channles
        self.high_channels = self.filters - self.low_channels
        
    def build(self, input_shape):
        assert len(input_shape) == 2
        assert len(input_shape[0]) == 4 and len(input_shape[1]) == 4
        # Assertion for high inputs
        assert input_shape[0][1] // 2 >= self.kernel_size[0]
        assert input_shape[0][2] // 2 >= self.kernel_size[1]
        # Assertion for low inputs
        assert input_shape[0][1] // input_shape[1][1] == 2
        assert input_shape[0][2] // input_shape[1][2] == 2
        # channels last for TensorFlow
        assert K.image_data_format() == "channels_last"
        # input channels
        high_in = int(input_shape[0][3])
        low_in = int(input_shape[1][3])

        # High -> High
        self.high_to_high_kernel = self.add_weight(name="high_to_high_kernel", 
                                    shape=(*self.kernel_size, high_in, self.high_channels),
                                    initializer=self.kernel_initializer,
                                    regularizer=self.kernel_regularizer,
                                    constraint=self.kernel_constraint)
        # High -> Low
        self.high_to_low_kernel  = self.add_weight(name="high_to_low_kernel", 
                                    shape=(*self.kernel_size, high_in, self.low_channels),
                                    initializer=self.kernel_initializer,
                                    regularizer=self.kernel_regularizer,
                                    constraint=self.kernel_constraint)
        # Low -> High
        self.low_to_high_kernel  = self.add_weight(name="low_to_high_kernel", 
                                    shape=(*self.kernel_size, low_in, self.high_channels),
                                    initializer=self.kernel_initializer,
                                    regularizer=self.kernel_regularizer,
                                    constraint=self.kernel_constraint)
        # Low -> Low
        self.low_to_low_kernel   = self.add_weight(name="low_to_low_kernel", 
                                    shape=(*self.kernel_size, low_in, self.low_channels),
                                    initializer=self.kernel_initializer,
                                    regularizer=self.kernel_regularizer,
                                    constraint=self.kernel_constraint)
        super().build(input_shape)

    def call(self, inputs):
        # Input = [X^H, X^L]
        assert len(inputs) == 2
        high_input, low_input = inputs
        # High -> High conv
        high_to_high = K.conv2d(high_input, self.high_to_high_kernel,
                                strides=self.strides, padding=self.padding,
                                data_format="channels_last")
        # High -> Low conv
        high_to_low  = K.pool2d(high_input, (2,2), strides=(2,2), pool_mode="avg")
        high_to_low  = K.conv2d(high_to_low, self.high_to_low_kernel,
                                strides=self.strides, padding=self.padding,
                                data_format="channels_last")
        # Low -> High conv
        low_to_high  = K.conv2d(low_input, self.low_to_high_kernel,
                                strides=self.strides, padding=self.padding,
                                data_format="channels_last")
        low_to_high = K.repeat_elements(low_to_high, 2, axis=1) # Nearest Neighbor Upsampling
        low_to_high = K.repeat_elements(low_to_high, 2, axis=2)
        # Low -> Low conv
        low_to_low   = K.conv2d(low_input, self.low_to_low_kernel,
                                strides=self.strides, padding=self.padding,
                                data_format="channels_last")
        # Cross Add
        high_add = high_to_high + low_to_high
        low_add = high_to_low + low_to_low
        return [high_add, low_add]

    def compute_output_shape(self, input_shapes):
        high_in_shape, low_in_shape = input_shapes
        high_out_shape = (*high_in_shape[:3], self.high_channels)
        low_out_shape = (*low_in_shape[:3], self.low_channels)
        return [high_out_shape, low_out_shape]

    def get_config(self):
        base_config = super().get_config()
        out_config = {
            **base_config,
            "filters": self.filters,
            "alpha": self.alpha,
            "filters": self.filters,
            "kernel_size": self.kernel_size,
            "strides": self.strides,
            "padding": self.padding,
            "kernel_initializer": self.kernel_initializer,
            "kernel_regularizer": self.kernel_regularizer,
            "kernel_constraint": self.kernel_constraint,            
        }
        return out_config
class OctConv2DTranspose(layers.Layer):
    def __init__(self, filters, alpha, kernel_size=(3, 3), strides=(2, 2), padding="same", kernel_initializer='glorot_uniform', kernel_regularizer=None, kernel_constraint=None, **kwargs):
        super(OctConv2DTranspose, self).__init__(**kwargs)
        self.alpha = alpha
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.kernel_initializer = kernel_initializer
        self.kernel_regularizer = kernel_regularizer
        self.kernel_constraint = kernel_constraint
        self.low_channels = int(filters * alpha)
        self.high_channels = filters - self.low_channels

    def build(self, input_shape):
        high_in_channels = input_shape[0][-1]
        low_in_channels = input_shape[1][-1]

        self.high_to_high = Conv2DTranspose(self.high_channels, self.kernel_size, strides=self.strides, padding=self.padding, kernel_initializer=self.kernel_initializer, kernel_regularizer=self.kernel_regularizer, kernel_constraint=self.kernel_constraint)

        self.low_to_high = Conv2DTranspose(self.high_channels, self.kernel_size, strides=self.strides, padding=self.padding, kernel_initializer=self.kernel_initializer, kernel_regularizer=self.kernel_regularizer, kernel_constraint=self.kernel_constraint)

        self.high_to_low = Conv2DTranspose(self.low_channels, self.kernel_size, strides=self.strides, padding=self.padding, kernel_initializer=self.kernel_initializer, kernel_regularizer=self.kernel_regularizer, kernel_constraint=self.kernel_constraint, use_bias=False)

        self.low_to_low = Conv2DTranspose(self.low_channels, self.kernel_size, strides=self.strides, padding=self.padding, kernel_initializer=self.kernel_initializer, kernel_regularizer=self.kernel_regularizer, kernel_constraint=self.kernel_constraint)

        self.up_sampling = UpSampling2D((2, 2), interpolation='nearest')

        super(OctConv2DTranspose, self).build(input_shape)

    def call(self, inputs):
        high_input, low_input = inputs

        # 고해상도 -> 고해상도 변환
        high_to_high = self.high_to_high(high_input)

        # 저해상도 -> 고해상도 변환 및 업샘플링
        low_to_high = self.low_to_high(low_input)
        low_to_high_upsampled = UpSampling2D(size=(2, 2))(low_to_high)

        # 저해상도 -> 저해상도 변환
        low_to_low = self.low_to_low(low_input)

        # 고해상도 -> 저해상도 변환 및 업샘플링
        high_to_low = self.high_to_low(AvgPool2D(pool_size=(2, 2))(high_input))
        high_to_low_upsampled = UpSampling2D(size=(2, 2))(high_to_low)

        # Cropping으로 high_to_low_upsampled의 크기 조정
        target_height, target_width = K.int_shape(low_to_low)[1:3]
        cropped_high_to_low = Cropping2D(((0, high_to_low_upsampled.shape[1] - target_height), (0, high_to_low_upsampled.shape[2] - target_width)))(high_to_low_upsampled)

        # 두 텐서의 크기를 조정한 후 Add 연산
        high_add = Add()([high_to_high, low_to_high_upsampled])

        # 조정된 크기로 low_add 계산
        low_add = Add()([cropped_high_to_low, low_to_low])

        return [high_add, low_add]

    def compute_output_shape(self, input_shape):
        high_shape, low_shape = input_shape
        high_output_shape = (high_shape[0], high_shape[1] * self.strides[0], high_shape[2] * self.strides[1], self.high_channels)
        low_output_shape = (low_shape[0], low_shape[1] * self.strides[0], low_shape[2] * self.strides[1], self.low_channels)
        return [high_output_shape, low_output_shape]

    def get_config(self):
        config = super(OctConv2DTranspose, self).get_config()
        config.update({
            "alpha": self.alpha,
            "filters": self.filters,
            "kernel_size": self.kernel_size,
            "strides": self.strides,
            "padding": self.padding,
            "kernel_initializer": self.kernel_initializer,
            "kernel_regularizer": self.kernel_regularizer,
            "kernel_constraint": self.kernel_constraint
        })
        return config
    
def contract_conv(input_x, filters, alpha, pad='same'):
    high, low = input_x
    high, low  = OctConv2D(filters, alpha,(3,3), padding=pad)([high, low])
    high = BatchNormalization()(high)
    low = BatchNormalization()(low)
    high = Dropout(0.3)(high)
    low = Dropout(0.3)(low)
    high, low = OctConv2D(filters, alpha,(3,3), padding=pad)([high, low])
    high = BatchNormalization()(high)
    low = BatchNormalization()(low)
    return [high, low]

filters, num_layers= 64, 4
input_layer = Input((256, 256, 3))  # 입력 차원 변경: 높이, 너비, 채널
low = AvgPool2D((2,2))(input_layer)  # 저주파 입력 레이어
unet_layers = [input_layer, low]
alpha = [0.5, 0.5, 0.25, 0.25]
conv_layers = []

for l in range(num_layers):
    unet_layers = contract_conv(unet_layers, filters=filters, alpha=alpha[l])
    conv_layers.append(unet_layers)
    unet_layers[0] = AveragePooling2D(2)(unet_layers[0])
    unet_layers[1] = AveragePooling2D(2)(unet_layers[1]) 
    filters *= 2
    
    
    
unet_layers = contract_conv(input_x=unet_layers, filters=filters, alpha=0.25,pad='same')
unet_layers[0] = AveragePooling2D(2)(unet_layers[0])
unet_layers[1] = AveragePooling2D(2)(unet_layers[1]) 
alpha_rev = [0.25, 0.25, 0.5, 0.5]
i = 0

for conv in reversed(conv_layers):
    filters //= 2
    unet_layers = OctConv2DTranspose(filters, alpha_rev[i],(2, 2),\
                                     strides=(2, 2), padding='same')(unet_layers)
    unet_layers[0] = concatenate([unet_layers[0], conv[0]])
    unet_layers[1] = concatenate([unet_layers[1], conv[1]])
    unet_layers = contract_conv(unet_layers, filters, alpha_rev[i])
    i+=1
    
ch_high = int(unet_layers[0].shape[-1])
h2h = Conv2DTranspose(ch_high, 3, strides=(2,2), padding='same')(unet_layers[0])
l2h = Conv2DTranspose(ch_high, 3, strides=(2,2), padding='same')(unet_layers[1])
l2h = Lambda(lambda x: K.repeat_elements(K.repeat_elements(x, 2, axis=1), 2, axis=2))(l2h)
x = Add()([h2h, l2h])
output_layer = Conv2D(1, (1,1), activation='sigmoid')(x)
print(output_layer)


OctaveUNnet = Model(inputs=input_layer, outputs=output_layer)
# OctaveUNnet.summary()     
def get_model(model_name, input_height=256, input_width=256, n_filters=64, n_channels=3, alpha=[0.5, 0.5, 0.25, 0.25], dropout_rate=0.3):
    if model_name == 'OctaveUNet':
        input_layer = Input((input_height, input_width, n_channels))
        low = AvgPool2D((2,2))(input_layer)
        unet_layers = [input_layer, low]
        conv_layers = []

        num_layers = 4
        for l in range(num_layers):
            unet_layers = contract_conv(unet_layers, filters=n_filters, alpha=alpha[l], pad='same')
            conv_layers.append(unet_layers)
            unet_layers[0] = AveragePooling2D(2)(unet_layers[0])
            unet_layers[1] = AveragePooling2D(2)(unet_layers[1])
            n_filters *= 2
        
        unet_layers = contract_conv(input_x=unet_layers, filters=n_filters, alpha=0.25, pad='same')
        unet_layers[0] = AveragePooling2D(2)(unet_layers[0])
        unet_layers[1] = AveragePooling2D(2)(unet_layers[1])
        alpha_rev = [0.25, 0.25, 0.5, 0.5]
        i = 0

        for conv in reversed(conv_layers):
            n_filters //= 2
            unet_layers = OctConv2DTranspose(n_filters, alpha_rev[i], (2, 2), strides=(2, 2), padding='same')(unet_layers)
            unet_layers[0] = Concatenate()([unet_layers[0], conv[0]])
            unet_layers[1] = Concatenate()([unet_layers[1], conv[1]])
            unet_layers = contract_conv(unet_layers, n_filters, alpha_rev[i])
            i += 1

        ch_high = int(unet_layers[0].shape[-1])
        h2h = Conv2DTranspose(ch_high, (3, 3), strides=(2, 2), padding='same')(unet_layers[0])
        l2h = Conv2DTranspose(ch_high, (3, 3), strides=(2, 2), padding='same')(unet_layers[1])
        l2h = Lambda(lambda x: K.repeat_elements(K.repeat_elements(x, 2, axis=1), 2, axis=2))(l2h)
        x = Add()([h2h, l2h])
        output_layer = Conv2D(1, (1, 1), activation='sigmoid')(x)

        model = Model(inputs=input_layer, outputs=output_layer)
        return model
    else:
        raise ValueError("Model name not recognized")

# 예제 사용
model = get_model('OctaveUNet', input_height=256, input_width=256, n_filters=64, n_channels=3)
model.summary()



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
BATCH_SIZE = 32   # batch size 지정
IMAGE_SIZE = (256, 256) # 이미지 크기 지정
MODEL_NAME = 'OctaveUNet' # 모델 이름
RANDOM_STATE = 3 # seed 고정
INITIAL_EPOCH = 0 # 초기 epoch
THESHOLDS = 0.25
lr = 0.01

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
EARLY_STOP_PATIENCE = 40

# 중간 가중치 저장 이름
CHECKPOINT_PERIOD = 5
CHECKPOINT_MODEL_NAME = 'checkpoint-{}-{}-epoch_{{epoch:02d}}_03_19_01.hdf5'.format(MODEL_NAME, save_name)
 
# 최종 가중치 저장 이름
FINAL_WEIGHTS_OUTPUT = 'model_{}_{}_final_weights_03_19_01.h5'.format(MODEL_NAME, save_name)

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
#  loss = sm.losses.binary_focal_jaccard_loss
# model 불러오기
model = get_model(MODEL_NAME, input_height=IMAGE_SIZE[0], input_width=IMAGE_SIZE[1], n_filters=N_FILTERS, n_channels=N_CHANNELS)
model.compile(optimizer = Adam(learning_rate=lr), loss = sm.losses.binary_focal_dice_loss, metrics = ['accuracy', miou])
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

model.load_weights('C:\\_data\\AI factory\\train_output\\model_attention_unet_base_line_final_weights_70_03_19_01.hdf5')


y_pred_dict = {}

for i in test_meta['test_img']:
    img = get_img_762bands(f'C:\\_data\\AI factory\\test_img\\{i}')
    y_pred = model.predict(np.array([img]), batch_size=1, verbose=1)
    
    y_pred = np.where(y_pred[0, :, :, 0] > 0.25, 1, 0) # 임계값 처리
    y_pred = y_pred.astype(np.uint8)
    y_pred_dict[i] = y_pred

joblib.dump(y_pred_dict, 'C:\\_data\\AI factory\\train_output\\y_pred_03_19_10.pkl')            