from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, Conv2DTranspose, Activation, multiply, add, BatchNormalization, Dropout
from keras.models import Model
from keras import backend as K
def conv2d_block(input_tensor, n_filters, kernel_size=3, batchnorm=True):
   
    # first layer
    x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), padding="same")(input_tensor)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation("relu")(x)

    # second layer
    x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), padding="same")(x)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation("relu")(x)
    return x

def attention_block(g, x, n_filters):
   
    Fg = Conv2D(n_filters, kernel_size=1, strides=1, padding='same')(g)  # gating signal
    Fx = Conv2D(n_filters, kernel_size=1, strides=1, padding='same')(x)  # image signal
    F = Activation('relu')(add([Fg, Fx]))
    psi = Conv2D(1, kernel_size=1, strides=1, padding='same', activation='sigmoid')(F)
    return multiply([x, psi])

def AttentionUNet(input_size=(256, 256, 3), n_filters=16, dropout=0.1, batchnorm=True):
    inputs = Input(input_size)
    n_filters = [n_filters, n_filters*2, n_filters*4, n_filters*8, n_filters*16]

    # Contracting Path
    c1 = conv2d_block(inputs, n_filters=n_filters[0], kernel_size=3, batchnorm=batchnorm)
    p1 = MaxPooling2D((2, 2))(c1)
    p1 = Dropout(dropout)(p1)

    c2 = conv2d_block(p1, n_filters=n_filters[1], kernel_size=3, batchnorm=batchnorm)
    p2 = MaxPooling2D((2, 2))(c2)
    p2 = Dropout(dropout)(p2)

    c3 = conv2d_block(p2, n_filters=n_filters[2], kernel_size=3, batchnorm=batchnorm)
    p3 = MaxPooling2D((2, 2))(c3)
    p3 = Dropout(dropout)(p3)

    c4 = conv2d_block(p3, n_filters=n_filters[3], kernel_size=3, batchnorm=batchnorm)
    p4 = MaxPooling2D(pool_size=(2, 2))(c4)
    p4 = Dropout(dropout)(p4)

    c5 = conv2d_block(p4, n_filters=n_filters[4], kernel_size=3, batchnorm=batchnorm)

    # Expansive Path
    u6 = Conv2DTranspose(n_filters[3], (3, 3), strides=(2, 2), padding='same')(c5)
    u6 = concatenate([u6, c4])
    u6 = Dropout(dropout)(u6)
    c6 = conv2d_block(u6, n_filters=n_filters[3], kernel_size=3, batchnorm=batchnorm)
    c6 = attention_block(g=c6, x=c4, n_filters=n_filters[3])

    u7 = Conv2DTranspose(n_filters[2], (3, 3), strides=(2, 2), padding='same')(c6)
    u7 = concatenate([u7, c3])
    u7 = Dropout(dropout)(u7)
    c7 = conv2d_block(u7, n_filters=n_filters[2], kernel_size=3, batchnorm=batchnorm)
    c7 = attention_block(g=c7, x=c3, n_filters=n_filters[2])

    u8 = Conv2DTranspose(n_filters[1], (3, 3), strides=(2, 2), padding='same')(c7)
    u8 = concatenate([u8, c2])
    u8 = Dropout(dropout)(u8)
    c8 = conv2d_block(u8, n_filters=n_filters[1], kernel_size=3, batchnorm=batchnorm)
    c8 = attention_block(g=c8, x=c2, n_filters=n_filters[1])

    u9 = Conv2DTranspose(n_filters[0], (3, 3), strides=(2, 2), padding='same')(c8)
    u9 = concatenate([u9, c1], axis=3)
    u9 = Dropout(dropout)(u9)
    c9 = conv2d_block(u9, n_filters=n_filters[0], kernel_size=3, batchnorm=batchnorm)
    c9 = attention_block(g=c9, x=c1, n_filters=n_filters[0])

    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c9)
    model = Model(inputs=[inputs], outputs=[outputs])

    return model

# Attention U-Net 모델 생성
model = AttentionUNet()
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()