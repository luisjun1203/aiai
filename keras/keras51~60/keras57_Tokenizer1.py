from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import numpy as np

text = "나는 진짜 진짜 매우 매우 맛있는 밥을 엄청 마구 마구 마구 먹었다"

token = Tokenizer()
token.fit_on_texts([text])

# print(token.word_index)
# {'마구': 1, '진짜': 2, '매우': 3, '나는': 4, '맛있는': 5, '밥을': 6, '엄청': 7, '먹었다': 8}    
# print(token.word_counts)
# OrderedDict([('나는', 1), ('진짜', 2), ('매우', 2), ('맛있는', 1), ('밥을', 1), ('엄청', 1), ('마구', 3), ('먹었다', 1)])

X = token.texts_to_sequences([text])
# print(X)
# [[4, 2, 2, 3, 3, 5, 6, 7, 1, 1, 1, 8]]

# [[[0. 0. 0. 0. 1. 0. 0. 0. 0.]
#   [0. 0. 1. 0. 0. 0. 0. 0. 0.]
#   [0. 0. 1. 0. 0. 0. 0. 0. 0.]
#   [0. 0. 0. 1. 0. 0. 0. 0. 0.]
#   [0. 0. 0. 1. 0. 0. 0. 0. 0.]
#   [0. 0. 0. 0. 0. 1. 0. 0. 0.]
#   [0. 0. 0. 0. 0. 0. 1. 0. 0.]
#   [0. 0. 0. 0. 0. 0. 0. 1. 0.]
#   [0. 1. 0. 0. 0. 0. 0. 0. 0.]
#   [0. 1. 0. 0. 0. 0. 0. 0. 0.]
#   [0. 1. 0. 0. 0. 0. 0. 0. 0.]
#   [0. 0. 0. 0. 0. 0. 0. 0. 1.]]]

# 1. to_categorical

# X1 = to_categorical(X)
# X1 = X1[:,:,1:]
# print(X1)
# [[[0. 0. 0. 1. 0. 0. 0. 0.]
#   [0. 1. 0. 0. 0. 0. 0. 0.]
#   [0. 1. 0. 0. 0. 0. 0. 0.]
#   [0. 0. 1. 0. 0. 0. 0. 0.]
#   [0. 0. 1. 0. 0. 0. 0. 0.]
#   [0. 0. 0. 0. 1. 0. 0. 0.]
#   [0. 0. 0. 0. 0. 1. 0. 0.]
#   [0. 0. 0. 0. 0. 0. 1. 0.]
#   [1. 0. 0. 0. 0. 0. 0. 0.]
#   [1. 0. 0. 0. 0. 0. 0. 0.]
#   [1. 0. 0. 0. 0. 0. 0. 0.]
#   [0. 0. 0. 0. 0. 0. 0. 1.]]]
# print(X1.shape)     # (1, 12, 8)


# 2. 사이킷 런 원핫인코더
# X = np.array(X)
# X = X.reshape(-1,1)
# ohe = OneHotEncoder(sparse=False)
# X2 = ohe.fit_transform(X)
# print(X2)
# [[0. 0. 0. 1. 0. 0. 0. 0.]
#  [0. 1. 0. 0. 0. 0. 0. 0.]
#  [0. 1. 0. 0. 0. 0. 0. 0.]
#  [0. 0. 1. 0. 0. 0. 0. 0.]
#  [0. 0. 1. 0. 0. 0. 0. 0.]
#  [0. 0. 0. 0. 1. 0. 0. 0.]
#  [0. 0. 0. 0. 0. 1. 0. 0.]
#  [0. 0. 0. 0. 0. 0. 1. 0.]
#  [1. 0. 0. 0. 0. 0. 0. 0.]
#  [1. 0. 0. 0. 0. 0. 0. 0.]
#  [1. 0. 0. 0. 0. 0. 0. 0.]
#  [0. 0. 0. 0. 0. 0. 0. 1.]]
# print(X2.shape)     #(12, 8)

# 3. 판다스 get dummies
X = np.array(X)
X = X.reshape(-1)
X3 = pd.get_dummies(X, dtype=int) 
                                    
print(X3)
#     1  2  3  4  5  6  7  8
# 0   0  0  0  1  0  0  0  0
# 1   0  1  0  0  0  0  0  0
# 2   0  1  0  0  0  0  0  0
# 3   0  0  1  0  0  0  0  0
# 4   0  0  1  0  0  0  0  0
# 5   0  0  0  0  1  0  0  0
# 6   0  0  0  0  0  1  0  0
# 7   0  0  0  0  0  0  1  0
# 8   1  0  0  0  0  0  0  0
# 9   1  0  0  0  0  0  0  0
# 10  1  0  0  0  0  0  0  0
# 11  0  0  0  0  0  0  0  1
print(X3.shape)     #(12, 8) 




