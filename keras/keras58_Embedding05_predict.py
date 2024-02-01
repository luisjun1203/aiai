from keras.preprocessing.text import Tokenizer
import numpy as np
import pandas as pd
from keras.utils import pad_sequences
# from keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from keras.models import Sequential, Model
from keras.layers import Embedding, LSTM, Dense, GRU, Conv1D, Conv2D, Dropout, Flatten, SimpleRNN
import numpy as np
import pandas as pd
from keras. callbacks import EarlyStopping, ModelCheckpoint
from keras. utils import to_categorical
from sklearn.model_selection import train_test_split 
from sklearn.metrics import f1_score,accuracy_score
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler
import time

# 1. 데이터
docs=[
    '너무 재미있다', '참 최고에요', '참 잘만든  영화에요',
    '추천하고 싶은 영화입니다.', '한 번 더 보고 싶어요.', '글쎄',
    '별로에요', '생각보다 지루해요', '연기가 어색해요',
    '재미없어요', '너무 재미없다.', '참 재밋네요.',
    '상헌이 바보', '반장 못생겼다', '욱이 또 잔다']

labels = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0])

token = Tokenizer()
token.fit_on_texts(docs)
# print(token.word_index)
# {'참': 1, '너무': 2, '재미있다': 3, '최고에요': 4, '잘만든': 5,
#  '영화에요': 6, '추천하고': 7, '싶은': 8, '영화입니다': 9, '한': 10,
#  '번': 11, '더': 12, '보고': 13, '싶어요': 14, '글쎄': 15,
#  '별로에요': 16, '생각보다': 17, '지루해요': 18, '연기가': 19, '어색해요': 20,
#  '재미없어요': 21, '재미없다': 22, '재밋네요': 23, '상헌이': 24, '바보': 25,
#  '반장': 26, '못생겼다': 27, '욱이': 28, '또': 29, '잔다': 30}
# print(token.word_counts)
# OrderedDict([('너무', 2), ('재미있다', 1), ('참', 3), ('최고에요', 1), ('잘만든', 1),
#              ('영화에요', 1), ('추천하고', 1), ('싶은', 1), ('영화입니다', 1), ('한', 1),
#              ('번', 1), ('더', 1), ('보고', 1), ('싶어요', 1), ('글쎄', 1),
#              ('별로에요', 1), ('생각보다', 1), ('지루해요', 1), ('연기가', 1), ('어색해요', 1),
#              ('재미없어요', 1), ('재미없다', 1), ('재밋네요', 1), ('상헌이', 1), ('바보', 1),
#              ('반장', 1), ('못생겼다', 1), ('욱이', 1), ('또', 1), ('잔다', 1)])
a = ['나는 정룡이가 정말 싫다. 재미없다 너무 정말']

token.fit_on_texts(a)
X = token.texts_to_sequences(docs)

X_pad = pad_sequences(X,
                        #  padding='pre',
                        #  maxlen=5,
                        # truncating='post'    
                        )


# X = X_pad.reshape(-1, 5, 1)               # 변환안해도 되는구만
# ohe = OneHotEncoder(sparse=False)

# labels = labels.reshape(-1,1)
# y = ohe.fit_transform(labels)


 
word_size = len(token.word_index)+ 1
print(word_size) 


model= Sequential()


##################### Embedding1 ######################################
# model.add(Embedding(input_dim=31,               # 단어사전의 개수 + 1 : input_dim, 
#                     output_dim=100,              # 출력 노드 개수
# )
#                     # input_length=5)              # 단어의 길이, 연산하는데 의미가 없다 -> 연산식에 포함 X
#          # Embedding input shape : 2차원, Embedding output shape : 3차원 
#           ) 

################## Embedding2 ################################
model.add(Embedding(input_dim= word_size , output_dim = 100, input_length=5))       # input_length가 1, 5는 돌아가지만 2,3,4,6... 안돼
# input_dim = 31    # 디폴트
# input_dim = 20    # 단어사전의 개수보다 작을때 : 연산량 줄어든다, 단어사전에서 임의으로 빠진다 : 성능 저하 될지도? ### 다른사람들은 돌아가는데 난 왜 안돌아갈까###
# input_dim = 40    # 단어사전의 개수보다 많을떄 : 연산량 늘어난다, 임의의 랜덤 임베딩 생성 : 성능 저하 될지도?

################## Embedding3 #######################################
# model.add(Embedding(31, 100))       # 잘 돌아간다

################## Embedding4 #######################################
# model.add(Embedding(31, 100, 5))       # 에러
# model.add(Embedding(31, 100, input_length=5))       # 잘 돌아간다
print(np.unique(X_pad, return_counts=True))

model.add(LSTM(units=19 ))        # 4*19*(19 + 100 + 1)                                                  
model.add(Dense(9, activation='relu'))
model.add(Dense(21, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(21, activation='relu'))
model.add(Dense(1,activation='sigmoid'))
model.summary()

'''
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
es = EarlyStopping(monitor='loss', mode='min', patience=300, verbose=2, restore_best_weights=True)
model.fit(X_pad, labels, epochs=1000, batch_size=10,verbose=2, callbacks=[es])



results = model.evaluate(X_pad, labels)

# print('loss' , results[0])
# print('acc', results[1])




aaa = token.texts_to_sequences(a)
aaa_pad = pad_sequences(aaa, padding = 'pre', maxlen=5, truncating='pre')
print(aaa_pad)
y_predict = model.predict(aaa_pad)
y_predict = np.array(y_predict)
# y_predict = np.argmax(y_predict, axis=1)
# print(token.word_index)
# {'너무': 1, '참': 2, '재미없다': 3, '정말': 4,
#  '재미있다': 5, '최고에요': 6, '잘만든': 7, '영화에요': 8,
#  '추천하고': 9, '싶은': 10, '영화입니다': 11, '한': 12,
#  '번': 13, '더': 14, '보고': 15, '싶어요': 16,
#  '글쎄': 17, '별로에요': 18, '생각보다': 19, '지루해요': 20,
#  '연기가': 21, '어색해요': 22, '재미없어요': 23, '재밋네요': 24,
#  '상헌이': 25, '바보': 26, '반장': 27, '못생겼다': 28,
#  '욱이': 29, '또': 30, '잔다': 31, '나는': 32, '정룡이가': 33, '좋다': 34}
print(y_predict)
'''






















