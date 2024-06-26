import segmentation_models as sm
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
from keras.models import *
from keras.layers import *
import numpy as np
from keras import backend as K
from sklearn.model_selection import train_test_split 
import joblib
tf.random.set_seed(42)
from keras_unet_collection import models

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
    img = rasterio.open(path).read((7,6,5)).transpose((1, 2, 0))    
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
                
                
train_meta = pd.read_csv('C:\\_data\\AI factory\\train_meta.csv')
test_meta = pd.read_csv('C:\\_data\\AI factory\\test_meta.csv')


# 저장 이름
save_name = '123'

N_FILTERS = 16 # 필터수 지정
N_CHANNELS = 3 # channel 지정
EPOCHS = 1000 # 훈련 epoch 지정
BATCH_SIZE = 32 # batch size 지정
IMAGE_SIZE = (256, 256) # 이미지 크기 지정
MODEL_NAME = 'pretrained_attention_unet' # 모델 이름
RANDOM_STATE = 3 # seed 고정
INITIAL_EPOCH = 0 # 초기 epoch

# 데이터 위치
IMAGES_PATH = 'C:\\_data\\AI factory\\train_img\\'
MASKS_PATH = 'C:\\_data\\AI factory\\train_mask\\'

# 가중치 저장 위치
OUTPUT_DIR = 'C:\_data\AI factory\\train_output\\'
WORKERS = -2

# 조기종료
EARLY_STOP_PATIENCE = 40

# 중간 가중치 저장 이름
CHECKPOINT_PERIOD = 5
CHECKPOINT_MODEL_NAME = 'checkpoint-{}-{}-epoch_{{epoch:02d}}_03_23_01.hdf5'.format(MODEL_NAME, save_name)
 
# 최종 가중치 저장 이름
FINAL_WEIGHTS_OUTPUT = 'model_{}_{}_final_weights_03_23_01.h5'.format(MODEL_NAME, save_name)

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
x_tr, x_val = train_test_split(train_meta, test_size=0.15, random_state=RANDOM_STATE)

# train : val 지정 및 generator
images_train = [os.path.join(IMAGES_PATH, image) for image in x_tr['train_img'] ]
masks_train = [os.path.join(MASKS_PATH, mask) for mask in x_tr['train_mask'] ]

images_validation = [os.path.join(IMAGES_PATH, image) for image in x_val['train_img'] ]
masks_validation = [os.path.join(MASKS_PATH, mask) for mask in x_val['train_mask'] ]

train_generator = generator_from_lists(images_train, masks_train, batch_size=BATCH_SIZE, random_state=RANDOM_STATE, image_mode="762")
validation_generator = generator_from_lists(images_validation, masks_validation, batch_size=BATCH_SIZE, random_state=RANDOM_STATE, image_mode="762")
#############################################모델################################################

#Default Conv2D
def conv2d_block(input_tensor, n_filters, kernel_size = 3, batchnorm = True):
    # first layer
    x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), kernel_initializer="he_normal",
               padding="same")(input_tensor)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation("swish")(x)

    # second layer
    x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), kernel_initializer="he_normal",
               padding="same")(x)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation("swish")(x)
    return x

#Attention Gate
def attention_gate(F_g, F_l, inter_channel):
    """
    An attention gate.

    Arguments:
    - F_g: Gating signal typically from a coarser scale.
    - F_l: The feature map from the skip connection.
    - inter_channel: The number of channels/filters in the intermediate layer.
    """
    # Intermediate transformation on the gating signal
    W_g = Conv2D(inter_channel, kernel_size=1, strides=1, padding='same', kernel_initializer='he_normal')(F_g)
    W_g = BatchNormalization()(W_g)

    # Intermediate transformation on the skip connection feature map
    W_x = Conv2D(inter_channel, kernel_size=1, strides=1, padding='same', kernel_initializer='he_normal')(F_l)
    W_x = BatchNormalization()(W_x)

    # Combine the transformations
    psi = Activation('swish')(add([W_g, W_x]))
    psi = Conv2D(1, kernel_size=1, strides=1, padding='same', kernel_initializer='he_normal')(psi)
    psi = BatchNormalization()(psi)
    psi = Activation('sigmoid')(psi)

    # Apply the attention coefficients to the feature map from the skip connection
    return multiply([F_l, psi])

from keras.applications import VGG16

def mymodel(f):
    inp = Input(shape=(256,256,3))
    c1 = Conv2D(filters=f*1, kernel_size=(3, 3), padding='same',)(inp)
    c1 = BatchNormalization()(c1)
    c1 = Activation("swish")(c1)
    c1 = Conv2D(filters=f*1, kernel_size=(3, 3), padding='same', )(c1)
    c1 = BatchNormalization()(c1)
    c1 = Activation("swish")(c1)
    p1 = MaxPooling2D()(c1)
    c2 = Conv2D(filters=f*2, kernel_size=(3, 3), padding='same', )(p1)
    c2 = BatchNormalization()(c2)
    c2 = Activation("swish")(c2)
    c2 = Conv2D(filters=f*2, kernel_size=(3, 3), padding='same', )(c2)
    c2 = BatchNormalization()(c2)
    c2 = Activation("swish")(c2)
    p2 = MaxPooling2D()(c2)
    c3 = Conv2D(filters=f*4, kernel_size=(3, 3), padding='same', )(p2)
    c3 = BatchNormalization()(c3)
    c3 = Activation("swish")(c3)
    c3 = Conv2D(filters=f*4, kernel_size=(3, 3), padding='same', )(c3)
    c3 = BatchNormalization()(c3)
    c3 = Activation("swish")(c3)
    p3 = MaxPooling2D()(c3)
    c4 = Conv2D(filters=f*8, kernel_size=(3, 3), padding='same', )(p3)
    c4 = BatchNormalization()(c4)
    c4 = Activation("swish")(c4)
    c4 = Conv2D(filters=f*8, kernel_size=(3, 3), padding='same', )(c4)
    c4 = BatchNormalization()(c4)
    c4 = Activation("swish")(c4)
    p4 = MaxPooling2D()(c4)
    c5 = Conv2D(filters=f*16, kernel_size=(3, 3), padding='same', )(p4)
    c5 = BatchNormalization()(c5)
    c5 = Activation("swish")(c5)
    c5 = Conv2D(filters=f*16, kernel_size=(3, 3), padding='same', )(c5)
    c5 = BatchNormalization()(c5)
    c5 = Activation("swish")(c5)
    
    
    return model

class SwishActivation(Layer):
    def __init__(self, **kwargs):
        super(SwishActivation, self).__init__(**kwargs)

    def call(self, inputs):
        return inputs * K.sigmoid(inputs)

    def get_config(self):
        config = super().get_config()
        return config
def conv_block(X, filters, block):
    # Residual block with dilated convolutions
    # Add skip connection at last after doing convolution operation to input X

    b = 'block_' + str(block) + '_'
    f1, f2, f3 = filters
    X_skip = X
    # block_a
    X = Conv2D(filters=f1, kernel_size=(1, 1), dilation_rate=(1, 1),
               padding='same', kernel_initializer='he_normal', name=b + 'a')(X)
    X = BatchNormalization(name=b + 'batch_norm_a')(X)
    X = SwishActivation(name=b + 'swish_activation_a')(X)
    # block_b
    X = Conv2D(filters=f2, kernel_size=(3, 3), dilation_rate=(2, 2),
               padding='same', kernel_initializer='he_normal', name=b + 'b')(X)
    X = BatchNormalization(name=b + 'batch_norm_b')(X)
    X = SwishActivation(name=b + 'swish_activation_b')(X)
    # block_c
    X = Conv2D(filters=f3, kernel_size=(1, 1), dilation_rate=(1, 1),
               padding='same', kernel_initializer='he_normal', name=b + 'c')(X)
    X = BatchNormalization(name=b + 'batch_norm_c')(X)
    # skip_conv
    X_skip = Conv2D(filters=f3, kernel_size=(3, 3), padding='same', name=b + 'skip_conv')(X_skip)
    X_skip = BatchNormalization(name=b + 'batch_norm_skip_conv')(X_skip)
    # block_c + skip_conv
    X = Add(name=b + 'add')([X, X_skip])
    # X = ReLU(name=b + 'relu')(X)
    X = SwishActivation(name=b + 'swish_activation_add')(X)
    
    return X
def get_pretrained_attention_unet(input_height=256, input_width=256, nClasses=1, n_filters=16, dropout=0.5, batchnorm=True, n_channels=3):
    # base_model = VGG16(weights='imagenet', include_top=False, input_shape=(input_height, input_width, n_channels))
    # base_model.summary()
    # Define the inputs
    # inputs = base_model.input
    
    # # Use specific layers from the VGG16 model for skip connections
    # s1 = base_model.get_layer("block1_conv2").output
    # s2 = base_model.get_layer("block2_conv2").output
    # s3 = base_model.get_layer("block3_conv3").output
    # s4 = base_model.get_layer("block4_conv3").output
    # bridge = base_model.get_layer("block5_conv3").output
    inp = Input(shape=(256,256,3))
    c1 = conv_block(inp, [n_filters, n_filters, n_filters*2], '1')
    # c1 = Conv2D(filters=n_filters*1, kernel_size=(3, 3), padding='same',)(inp)
    # c1 = BatchNormalization()(c1)
    # c1 = Activation("swish")(c1)
    # c1 = Conv2D(filters=n_filters*1, kernel_size=(3, 3), padding='same',)(c1)
    # c1 = BatchNormalization()(c1)
    # c1 = Activation("swish")(c1)
    # c1 = Conv2D(filters=n_filters*1, kernel_size=(3, 3), padding='same', )(c1)
    # c1 = BatchNormalization()(c1)
    # c1 = Activation("swish")(c1)
    p1 = MaxPooling2D()(c1)
    c2 = conv_block(p1, [n_filters, n_filters, n_filters*2], '2')
    
    # c2 = Conv2D(filters=n_filters*2, kernel_size=(3, 3), padding='same', )(p1)
    # c2 = BatchNormalization()(c2)
    # c2 = Activation("swish")(c2)
    # c2 = Conv2D(filters=n_filters*2, kernel_size=(3, 3), padding='same', )(c2)
    # c2 = BatchNormalization()(c2)
    # c2 = Activation("swish")(c2)
    # c2 = Conv2D(filters=n_filters*2, kernel_size=(3, 3), padding='same', )(c2)
    # c2 = BatchNormalization()(c2)
    # c2 = Activation("swish")(c2)
    p2 = MaxPooling2D()(c2)
    c3 = conv_block(p2, [n_filters, n_filters, n_filters*2], '3')
    # c3 = Conv2D(filters=n_filters*4, kernel_size=(3, 3), padding='same', )(p2)
    # c3 = BatchNormalization()(c3)
    # c3 = Activation("swish")(c3)
    # c3 = Conv2D(filters=n_filters*4, kernel_size=(3, 3), padding='same', )(c3)
    # c3 = BatchNormalization()(c3)
    # c3 = Activation("swish")(c3)
    # c3 = Conv2D(filters=n_filters*4, kernel_size=(3, 3), padding='same', )(c3)
    # c3 = BatchNormalization()(c3)
    # c3 = Activation("swish")(c3)
    p3 = MaxPooling2D()(c3)
    c4 = conv_block(p3, [n_filters, n_filters, n_filters*2], '4')
    # c4 = Conv2D(filters=n_filters*8, kernel_size=(3, 3), padding='same', )(p3)
    # c4 = BatchNormalization()(c4)
    # c4 = Activation("swish")(c4)
    # c4 = Conv2D(filters=n_filters*8, kernel_size=(3, 3), padding='same', )(c4)
    # c4 = BatchNormalization()(c4)
    # c4 = Activation("swish")(c4)
    # c4 = Conv2D(filters=n_filters*8, kernel_size=(3, 3), padding='same', )(c4)
    # c4 = BatchNormalization()(c4)
    # c4 = Activation("swish")(c4)
    p4 = MaxPooling2D()(c4)
    bridge = conv_block(p4, [n_filters, n_filters, n_filters*2], '5')
    # c5 = Conv2D(filters=n_filters*16, kernel_size=(3, 3), padding='same', )(p4)
    # c5 = BatchNormalization()(c5)
    # c5 = Activation("swish")(c5)
    # c5 = Conv2D(filters=n_filters*16, kernel_size=(3, 3), padding='same', )(c5)
    # c5 = BatchNormalization()(c5)
    # c5 = Activation("swish")(c5)
    # c5 = Conv2D(filters=n_filters*16, kernel_size=(3, 3), padding='same', )(c5)
    # c5 = BatchNormalization()(c5)
    # bridge = Activation("swish")(c5)
    # Decoder with attention gates
    d1 = UpSampling2D((2, 2))(bridge)
    d1 = concatenate([d1, attention_gate(d1, c4, n_filters*8)])
    d1 = conv2d_block(d1, n_filters*10, kernel_size=3, batchnorm=batchnorm)
    # 풀링 한방씩 제거
    d2 = UpSampling2D((2, 2))(d1)
    d2 = concatenate([d2, attention_gate(d2, c3, n_filters*4)])
    d2 = conv2d_block(d2, n_filters*5, kernel_size=3, batchnorm=batchnorm)
    
    d3 = UpSampling2D((2, 2))(d2)
    d3 = concatenate([d3, attention_gate(d3, c2, n_filters*2)])
    d3 = conv2d_block(d3, n_filters*2, kernel_size=3, batchnorm=batchnorm)
    
    d4 = UpSampling2D((2, 2))(d3)
    d4 = concatenate([d4, attention_gate(d4, c1, n_filters)])
    d4 = conv2d_block(d4, n_filters, kernel_size=3, batchnorm=batchnorm)
    
    outputs = Conv2D(nClasses, (1, 1), activation='sigmoid')(d4)
    model = Model(inputs=[inp], outputs=[outputs])
    return model

def get_model(model_name, nClasses=1, input_height=256, input_width=256, n_filters = 16, dropout = 0.1, batchnorm = True, n_channels=10):
    
    if model_name == 'pretrained_attention_unet':
        model = get_pretrained_attention_unet
        
        
    return model(
            nClasses      = nClasses,
            input_height  = input_height,
            input_width   = input_width,
            n_filters     = n_filters,
            dropout       = dropout,
            batchnorm     = batchnorm,
            n_channels    = n_channels
        )
    
    
# model = get_model(MODEL_NAME, input_height=IMAGE_SIZE[0], input_width=IMAGE_SIZE[1], n_filters=N_FILTERS, n_channels=N_CHANNELS)
model = get_model(MODEL_NAME, input_height=256, input_width=256, n_filters=32, n_channels=N_CHANNELS)
# model = models.att_unet_2d(input_size=(256,256,3), filter_num=[64, 128, 256, 512], n_labels=1, activation='ReLU', output_activation='Sigmoid', batch_norm=True, backbone='VGG16', weights='imagenet',)
model.compile(
    'Adam',
    loss=sm.losses.binary_focal_dice_loss,
    metrics=[sm.metrics.iou_score],
)

model.summary()

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=EARLY_STOP_PATIENCE, restore_best_weights=True)
mcp = ModelCheckpoint(monitor='val_loss', mode='min', verbose=1, save_best_only=True, filepath=CHECKPOINT_MODEL_NAME)
rlr = ReduceLROnPlateau(
    monitor='val_loss',
    patience=7,
    verbose=1,
    factor=0.7,
    min_lr = 0.000001 
)

model.fit_generator(
    train_generator,
    steps_per_epoch=len(images_train) // BATCH_SIZE,
    validation_data=validation_generator,
    validation_steps=len(images_validation) // BATCH_SIZE,
    epochs=EPOCHS,
    workers=WORKERS,
    callbacks=[es,rlr,mcp],
)

print('가중치 저장')
model_weights_output = os.path.join(OUTPUT_DIR, FINAL_WEIGHTS_OUTPUT)
model.save_weights(model_weights_output)
print("저장된 가중치 명: {}".format(model_weights_output))


model.load_weights('C:\\_data\\AI factory\\train_output\\checkpoint-pretrained_attention_unet_123_final_weights_03_23_01.hdf5')


y_pred_dict = {}

for i in test_meta['test_img']:
    img = get_img_762bands(f'C:\\_data\\AI factory\\test_img\\{i}') # 나중에 여기도 preprocessing 해줘야되는데, 일단 가중치 저장하고 하자
    y_pred = model.predict(np.array([img]), batch_size=1)
    
    y_pred = np.where(y_pred[0, :, :, 0] > 0.25, 1, 0) # 임계값 처리
    y_pred = y_pred.astype(np.uint8)
    y_pred_dict[i] = y_pred

joblib.dump(y_pred_dict, 'C:\\_data\\AI factory\\train_output\\y_pred_03_23_01.pkl') 
print("done")