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

# def build_discriminator(shape=512,
#                         ndf=64, n_layers=3,
#                         kernel_size=4, strides=2, activation='linear',
#                         n_downsampling=1):
    
#     img_a_input_512 = Input(shape=(512,512,1))
#     img_b_input_512 = Input(shape=(512,512,3))

#     img_a_input_256 = Input(shape=(256,256,1))
#     img_b_input_256 = Input(shape=(256,256,3))


#     img_a_input_128 = Input(shape=(128,128,1))
#     img_b_input_128 = Input(shape=(128,128,3))

#     img_a_input_64 = Input(shape=(64,64,1))
#     img_b_input_64 = Input(shape=(64,64,3))
    
#     if shape==512:
#         input_a=img_a_input_512
#         input_b=img_b_input_512
        
    
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
#     model = Model(inputs=[img_a_input_512,img_a_input_256,img_a_input_128,img_a_input_64, input_b], outputs=[x] + features)
#     print("\nDiscriminator")
#     # model.summary()
#     return model


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
	g = Activation('relu')(g)
	return g


# define an encoder block
def define_encoder_block(layer_in, n_filters, batchnorm=True):
	# weight initialization
	init = RandomNormal(stddev=0.02)
	# add downsampling layer
	g = Conv2D(n_filters, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(layer_in)
	# conditionally add batch normalization
	if batchnorm:
		g = BatchNormalization()(g, training=True)
	# leaky relu activation
	g = LeakyReLU(alpha=0.2)(g)
	return g

# define a decoder block
def decoder_block(layer_in, skip_in, n_filters, dropout=True):
	# weight initialization
	init = RandomNormal(stddev=0.02)
	# add upsampling layer
	g = Conv2DTranspose(n_filters, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(layer_in)
	# add batch normalization
	g = BatchNormalization()(g, training=True)
	# conditionally add dropout
	if dropout:
		g = Dropout(0.5)(g, training=True)
	# merge with skip connection
	g = Concatenate()([g, skip_in])
	# relu activation
	g = Activation('relu')(g)
	return g





# define the standalone generator model
def define_generator_1024(image_shape=(1024,1024,3)):
	# weight initialization
	init = RandomNormal(stddev=0.02)
    
#.........................................512..........................

	image_shape_512 = (int(image_shape[0]/2),int(image_shape[1]/2),int(image_shape[2]))
    
	in_image_512 = Input(shape=image_shape_512) 
    
# #.................................

# 	e512_special = define_normal_encoder_block(in_image_512, 16)
#...........................
    #concatenate
        
	e_512_1=define_normal_encoder_block(in_image_512, 32)
	e_512_middle=define_normal_encoder_block(e_512_1,32)
	e_512_2=define_encoder_block(e_512_middle,32)   


#.........................................256..........................

	image_shape_256 = (int(image_shape[0]/4),int(image_shape[1]/4),int(image_shape[2]))
    
	in_image_256 = Input(shape=image_shape_256)
    
	e256_special = define_normal_encoder_block(in_image_256, 32)
    
# #...........................
#     #concatenate
    
	concatenate_512_256 = Concatenate()([e256_special,e_512_2])   
    
	e_256_1=define_normal_encoder_block(concatenate_512_256, 64)
	e_256_middle = define_normal_encoder_block(e_256_1,64)
	e_256_2=define_encoder_block(e_256_middle,64)  
    

    

#.............................................128...............
	image_shape_128 = (int(image_shape[0]/8),int(image_shape[1]/8),int(image_shape[2]))
    
	in_image_128 = Input(shape=image_shape_128)
    
	e128_special = define_normal_encoder_block(in_image_128, 64)
    
	concatenate_256_128 = Concatenate()([e128_special,e_256_2]) 

    
	e_128_1=define_normal_encoder_block(concatenate_256_128, 128)
	e_128_middle=define_normal_encoder_block(e_128_1, 128)
	e_128_2=define_encoder_block(e_128_middle,128)

    
#..............................................64..................
	image_shape_64 =(int(image_shape[0]/16),int(image_shape[1]/16),int(image_shape[2]))

	# image input
	in_image_64 = Input(shape=image_shape_64)
	# encoder model
	e1 = define_normal_encoder_block(in_image_64, 128, batchnorm=False)
#...........................
    #concatenate
    
	concatenate_128_64 = Concatenate()([e1,e_128_2])

#.............................
    
	e2 = define_normal_encoder_block(concatenate_128_64, 256)
	e3 = define_encoder_block(e2, 256)
	e4 = define_encoder_block(e3, 512)
	e5 = define_encoder_block(e4, 512)
	e6 = define_encoder_block(e5, 512)
	e7 = define_encoder_block(e6, 512)
	# bottleneck, no batch norm and relu
	b = Conv2D(512, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(e7)
	b = Activation('relu')(b)
	# decoder model
	d1 = decoder_block(b, e7, 512)
	d2 = decoder_block(d1, e6, 512)
	d3 = decoder_block(d2, e5, 512)
	d4 = decoder_block(d3, e4, 512, dropout=False)
	d5 = decoder_block(d4, e3, 256, dropout=False)
	d6 = decoder_block(d5, e2, 256, dropout=False)
 
	d7 = normal_decoder_block(d6, e1, 128, dropout=False)
# 	d_attention_128=Attention(256)(d7)
    
##....................64 output
	# output
	g = Conv2D(3, (4,4), strides=(1,1), padding='same', kernel_initializer=init)(d7)
	out_image = Activation('tanh')(g)
    
##...............................128 output

	d8 = decoder_block(d7, e_128_middle, 128, dropout=False)
	d9 = normal_decoder_block(d8, e_128_1, 128, dropout=False)
	d10 = normal_decoder_block(d9, e128_special, 64, dropout=False)
	# output
	# output

	g_128 = Conv2D(3, (4,4), strides=(1,1), padding='same', kernel_initializer=init)(d10)
	out_image_128 = Activation('tanh')(g_128)
    
# ##...............................256 output

	d11 = decoder_block(d10, e_256_middle, 64, dropout=False)
# 	d_attention_256=Attention(128)(d11)
	d12 = normal_decoder_block(d11, e_256_1, 64, dropout=False)
# 	# output
	g_256 = Conv2D(3, (4,4), strides=(1,1), padding='same', kernel_initializer=init)(d12)
	out_image_256 = Activation('tanh')(g_256)
    
####....................................512 output

	d13 = decoder_block(d12, e_512_middle, 32, dropout=False)
# 	d_attention_512=Attention(64)(d13)
	d14 = normal_decoder_block(d13, e_512_1, 32, dropout=False)
# 	# output
	g_512 = Conv2D(3, (4,4), strides=(1,1), padding='same', kernel_initializer=init)(d14)
	out_image_512 = Activation('sigmoid')(g_512)
    
	# define model
	model = Model([in_image_512,in_image_256,in_image_128,in_image_64], [out_image_512,out_image_256,out_image_128,out_image])
# 	model = multi_gpu_model(model, gpus=2)
	return model
g_model = define_generator_1024(image_shape=(1024,1024,1))

g_model.summary()