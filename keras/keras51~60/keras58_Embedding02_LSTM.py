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

X = token.texts_to_sequences(docs)
X_pad = pad_sequences(X,
                        #  padding='pre',
                        #  maxlen=5,
                        # truncating='post'
                        
                        )
# print(X_pad)
# [[2, 3], [1, 4], [1, 5, 6],
#  [7, 8, 9], [10, 11, 12, 13, 14], [15],
#  [16], [17, 18], [19, 20],
#  [21], [2, 22], [1, 23],
#  [24, 25], [26, 27], [28, 29, 30]]

# print(type(X))  # <class 'list'>
# X = np.array(X) # 적용안된다 차원 다름

# X = X_pad.reshape(-1, 5, 1)               # 변환안해도 되는구만
ohe = OneHotEncoder(sparse=False)

labels = labels.reshape(-1,1)
y = ohe.fit_transform(labels)

# print(y)
# print(y.shape)          # (15, 2)
# print(X)     # (75, 31)
# print(X.shape)     # (75, 31)
 


model= Sequential()
model.add(LSTM(units=19, input_length=5, input_dim = 1))                                                      
model.add(Dense(9, activation='swish'))
model.add(Dense(21, activation='swish'))
model.add(Dense(16, activation='swish'))
model.add(Dense(21, activation='swish'))
model.add(Dense(2,activation='softmax'))
model.summary()

# strat_time = time.time()
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
es = EarlyStopping(monitor='val_loss', mode='min', patience=300, verbose=2, restore_best_weights=True)
model.fit(X_pad, y, epochs=1000, batch_size=50,verbose=2, callbacks=[es])
# end_time = time.time()


results = model.evaluate(X_pad, y)
# a = ['영화는 재미없다']

# token.fit_on_texts(a)

# aaa = token.texts_to_sequences(a)
# aaa_pad = pad_sequences(aaa, maxlen=5 )
# # y_predict = y_predict.reshape
# y_predict = model.predict(aaa)
# print(y_predict)


# y_predict = model.predict(a)

# y_test = np.argmax(y_test, axis=1)
# y_predict = np.argmax(y_predict, axis=1)
# fs = f1_score(y_test, y_predict, average='macro')
# # print(y_test)
print('loss' , results[0])
print('acc', results[1])
# print("걸리시간 : ", round(end_time - strat_time, 3), "초")
# print("f1_score : ", fs)

# loss 2.6820905986824073e-05
# acc 1.0



























