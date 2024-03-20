from keras.applications import VGG16
model = VGG16(weights='imagenet', include_top=False, input_shape=(256, 256, 3))
model.summary()