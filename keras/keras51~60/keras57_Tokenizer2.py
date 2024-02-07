from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from keras.preprocessing.sequence import pad_sequences
import pandas as pd
import numpy as np

text1 = "나는 진짜 진짜 매우 매우 맛있는 밥을 엄청 마구 마구 마구 먹었다"
text2 = "상헌이가 선생을 괴롭힌다. 상헌이는 못생겼다. 상헌이는 마구 마구 못생겼다."

#####아래 수정해봐############# 

token = Tokenizer()
token.fit_on_texts([text1, text2])

# print(token.word_index)
# {'마구': 1, '진짜': 2, '매우': 3, '상헌이는': 4, '못생겼다': 5, '나는': 6, '맛있는': 7, '밥을': 8, '엄청': 9, '먹었다': 10, '상헌이가': 11, '선생을': 12, '괴롭힌다': 13}  
# print(token.word_counts)
# [('나는', 1), ('진짜', 2), ('매우', 2), ('맛있는', 1), ('밥을', 1), ('엄청', 1), ('마구', 5), ('먹었다', 1), ('상헌이가', 1),
#  ('선생을', 1), ('괴롭힌다', 1), ('상헌이는', 2), ('못생겼다', 2)])



X = token.texts_to_sequences([text1, text2])
X_padded = pad_sequences(X, padding='pre',maxlen=12)



# print(X)
# [[6, 2, 2, 3, 3, 7, 8, 9, 1, 1, 1, 10], [11, 12, 13, 4, 5, 4, 1, 1, 5]]


# 1. to_categorical

# X1 = to_categorical(X_padded)
# X1 = X1[:,:-3,1:]
# print(X1)

# print(X1.shape)     # (2, 9, 13)


# 2. 사이킷 런 원핫인코더
# X = np.array(X_padded)
# X_padded = X.reshape(-1,1)
# ohe = OneHotEncoder(sparse=False)
# X2 = ohe.fit_transform(X_padded)
# X2 = X2[:, :]
# print(X2)


# print(X2.shape)     #(24, 14)

# 3. 판다스 get dummies
# X = np.array(X_padded)
# X_padded = X.reshape(-1)
# X3 = pd.get_dummies(X_padded, dtype=int) 
                                    
# print(X3)

# print(X3.shape)     #(24, 14)




