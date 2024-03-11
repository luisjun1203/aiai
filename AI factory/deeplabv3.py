import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Conv2D, BatchNormalization, Activation, UpSampling2D, Concatenate, DepthwiseConv2D
from keras.applications import MobileNetV2

def ASPP(inputs, filters=256):
    
    shape = inputs.shape
    
    y1 = Conv2D(filters, 1, padding="same")(inputs)
    y1 = BatchNormalization()(y1)
    y1 = Activation("relu")(y1)
    
    y2 = DepthwiseConv2D(3, dilation_rate=(6, 6), padding="same", use_bias=False)(inputs)
    y2 = BatchNormalization()(y2)
    y2 = Activation("relu")(y2)
    y2 = Conv2D(filters, 1, padding="same")(y2)
    y2 = BatchNormalization()(y2)
    y2 = Activation("relu")(y2)
    
    y3 = DepthwiseConv2D(3, dilation_rate=(12, 12), padding="same", use_bias=False)(inputs)
    y3 = BatchNormalization()(y3)
    y3 = Activation("relu")(y3)
    y3 = Conv2D(filters, 1, padding="same")(y3)
    y3 = BatchNormalization()(y3)
    y3 = Activation("relu")(y3)
    
    y4 = DepthwiseConv2D(3, dilation_rate=(18, 18), padding="same", use_bias=False)(inputs)
    y4 = BatchNormalization()(y4)
    y4 = Activation("relu")(y4)
    y4 = Conv2D(filters, 1, padding="same")(y4)
    y4 = BatchNormalization()(y4)
    y4 = Activation("relu")(y4)
    
    y5 = tf.keras.layers.AveragePooling2D(pool_size=(shape[1], shape[2]))(inputs)
    y5 = Conv2D(filters, 1, padding="same")(y5)
    y5 = BatchNormalization()(y5)
    y5 = Activation("relu")(y5)
    y5 = UpSampling2D(size=(shape[1], shape[2]), interpolation="bilinear")(y5)
    
    y = Concatenate()([y1, y2, y3, y4, y5])
    
    y = Conv2D(filters, 1, padding="same")(y)
    y = BatchNormalization()(y)
    y = Activation("relu")(y)
    
    return y

def DeepLabV3Plus(image_size, num_classes):
    base_model = MobileNetV2(input_shape=(image_size, image_size, 3), include_top=False)
    image_features = base_model.get_layer('block_13_expand_relu').output
    x_a = ASPP(image_features)
    x_a = UpSampling2D(size=(4, 4), interpolation="bilinear")(x_a)

    x_b = base_model.get_layer('block_3_expand_relu').output
    x_b = Conv2D(48, 1, padding="same")(x_b)
    x_b = BatchNormalization()(x_b)
    x_b = Activation("relu")(x_b)

    x = Concatenate()([x_a, x_b])
    x = Conv2D(256, 3, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv2D(256, 3, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv2D(num_classes, 1, padding="same")(x)
    x = UpSampling2D(size=(4, 4), interpolation="bilinear")(x)
    x = Activation("sigmoid")(x) 
    
    model = Model(inputs=base_model.input, outputs=x)
    return model

# 모델 생성
model = DeepLabV3Plus(image_size=256, num_classes=21)  # 21은 예시 클래스 수, 실제 클래스 수에 맞게 조정
model.summary()