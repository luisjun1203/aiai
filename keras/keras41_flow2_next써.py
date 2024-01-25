from keras.datasets import fashion_mnist
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt
from keras.utils.image_utils import img_to_array, load_img

(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

train_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    vertical_flip=True,
    width_shift_range=0.2,
    height_shift_range=0.2,
    rotation_range=30,
    zoom_range=0.2,
    shear_range=30,
    fill_mode='nearest'
    
)
augment_size = 100

print(X_train[0].shape)     # (28, 28)
# plt.imshow(X_train[0], 'gray')
# plt.show()

X_data = train_datagen.flow(            # np.tile(A, repeat_shape) :  A 배열이 repeat_shape 형태로 반복되어 쌓인 형태 -> 즉, 동일한 내용의 형태를 반복해서 붙여준다
    np.tile(X_train[0].reshape(28*28), augment_size).reshape(-1, 28, 28, 1),             # X (100,28,28,1)
    np.zeros(augment_size),                                                             # y  (100, ) # 0으로만 채워진 array 생성
    batch_size=50,
    shuffle=False,
    
    ).next()        # 다음 데이터가 없으니 .next()를 하면 X_data에 (100, 28, 28, 1)이 들어감
# print(X_data)
# print(X_data.shape) # 튜플형태라서 에러, 왜냐면 flow에서 튜플 형태로 반환해준다
print(X_data[0].shape)  # (100, 28, 28, 1)
print(X_data[1].shape)  # (100,) 전부다 0
print(np.unique(X_data[1],return_counts = True))    # (array([0.]), array([100], dtype=int64))

print(X_data[0][0].shape)      #  (28, 28, 1)


plt.figure(figsize=(7, 7))
for i in range(50):
    plt.subplot(10, 10, i+1)
    plt.axis('off')
    plt.imshow(X_data[0][i], cmap = 'gray')    
plt.show()




