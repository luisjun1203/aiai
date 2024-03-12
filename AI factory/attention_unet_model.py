import tensorflow as tf
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, Conv2DTranspose
from keras.layers import Activation, BatchNormalization, Dropout, Lambda, Multiply, Add
from keras.models import Model
from keras import backend as K

def conv2d_block(input_tensor, n_filters, kernel_size=3, batchnorm=True):
    """Function to add 2 convolutional layers with the parameters passed to it"""
    # first layer
    x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), kernel_initializer="he_normal", padding="same")(input_tensor)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation("relu")(x)

    # second layer
    x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), kernel_initializer="he_normal", padding="same")(x)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation("relu")(x)
    return x

def attention_gate(input_tensor, gating_signal, n_filters, kernel_size=1, batchnorm=True):
    """Function to add attention gate"""
    theta_x = Conv2D(n_filters, (kernel_size, kernel_size), strides=(2, 2), padding='same')(input_tensor)
    phi_g = Conv2D(n_filters, (kernel_size, kernel_size), padding='same')(gating_signal)

    concat = Add()([theta_x, phi_g])
    if batchnorm:
        concat = BatchNormalization()(concat)
    concat = Activation('relu')(concat)

    psi = Conv2D(1, (kernel_size, kernel_size), padding='same')(concat)
    psi = BatchNormalization()(psi)
    psi = Activation('sigmoid')(psi)

    upsample_psi = UpSampling2D(size=(2, 2))(psi)
    output = Multiply()([input_tensor, upsample_psi])
    return output

def encoder_block(input_tensor, n_filters, pool_size=(2,2), dropout=0.3, batchnorm=True):
    """Function to define an encoder block in Attention U-Net"""
    c = conv2d_block(input_tensor, n_filters, kernel_size=3, batchnorm=batchnorm)
    p = MaxPooling2D(pool_size)(c)
    p = Dropout(dropout)(p)
    return c, p

def decoder_block(input_tensor, concat_tensor, n_filters, dropout=0.3, batchnorm=True):
    """Function to define a decoder block in Attention U-Net"""
    u = Conv2DTranspose(n_filters, (3, 3), strides=(2, 2), padding='same')(input_tensor)
    c = concatenate([u, concat_tensor])
    c = Dropout(dropout)(c)
    c = conv2d_block(c, n_filters, kernel_size=3, batchnorm=batchnorm)
    return c

def AttentionUNet(input_shape=(256, 256, 1), n_filters=16, dropout=0.1, batchnorm=True):
    inputs = Input(input_shape)
    
    # Encoder pathway
    c1, p1 = encoder_block(inputs, n_filters * 1, batchnorm=batchnorm)
    c2, p2 = encoder_block(p1, n_filters * 2, batchnorm=batchnorm)
    c3, p3 = encoder_block(p2, n_filters * 4, batchnorm=batchnorm)
    c4, p4 = encoder_block(p3, n_filters * 8, batchnorm=batchnorm)
    
    # Bridge
    b = conv2d_block(p4, n_filters * 16, kernel_size=3, batchnorm=batchnorm)
    
    # Decoder pathway
    a1 = attention_gate(c4, b, n_filters * 8)
    d1 = decoder_block(b, a1, n_filters * 8, dropout=dropout, batchnorm=batchnorm)

    a2 = attention_gate(c3, d1, n_filters * 4)
    d2 = decoder_block(d1, a2, n_filters * 4, dropout=dropout, batchnorm=batchnorm)

    a3 = attention_gate(c2, d2, n_filters * 2)
    d3 = decoder_block(d2, a3, n_filters * 2, dropout=dropout, batchnorm=batchnorm)

    a4 = attention_gate(c1, d3, n_filters)
    d4 = decoder_block(d3, a4, n_filters, dropout=dropout, batchnorm=batchnorm)

    # Output
    outputs = Conv2D(1, (1, 1), activation='sigmoid')(d4)

    model = Model(inputs=[inputs], outputs=[outputs])
    return model

# Create model
model = AttentionUNet(input_shape=(256, 256, 3), n_filters=16, dropout=0.1, batchnorm=True)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()