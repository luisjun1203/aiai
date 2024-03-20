from keras.applications.mobilenet_v3 import MobileNetV3Small,MobileNetV3Large



model = MobileNetV3Small(weights='imagenet', include_top=False, input_shape=(256, 256, 3))

model.summary()