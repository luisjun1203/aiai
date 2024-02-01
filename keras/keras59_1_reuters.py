from keras.datasets import reuters
from keras.preprocessing.text import Tokenizer
import numpy as np
import pandas as pd
from keras.utils import pad_sequences
# from keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from keras.models import Sequential, Model
from keras.layers import Embedding, LSTM, Dense, GRU, Conv1D, Conv2D, Dropout, Flatten, SimpleRNN, Reshape, GlobalAveragePooling2D, MaxPooling2D
import numpy as np
import pandas as pd
from keras. callbacks import EarlyStopping, ModelCheckpoint
from keras. utils import to_categorical
from sklearn.model_selection import train_test_split 
from sklearn.metrics import f1_score,accuracy_score
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler
import time



(X_train,  y_train), (X_test, y_test) = reuters.load_data(num_words=2048,     # 단어사용 빈도수에 맞춰서 순서대로 설정가능, None은 모든 단어 전부다 사용
                                                          test_split=0.2
                                                          
                                                          )
# print(X_train.shape)    # (8982,)
# print(y_train.shape)    # (8982,)
# print(X_test.shape)     # (2246,)
# print(y_test.shape)     # (2246,)
# print(len(X_train[0]))  # 87
# print(len(X_train[1]))  # 56
print("뉴스기사의 최대길이 : ", max(len(i) for i in X_train))           # 2376
print("뉴스기사의 평균거리 : ", sum(map(len,X_train))/ len(X_train))    # 뉴스기사의 평균거리 :  145.5398574927633





# print(len(np.unique(y_train)))  # 46
# print(len(np.unique(y_test)))   # 46

# (array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,
#        17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33,
#        34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45], dtype=int64), array([  55,  432,   74, 3159, 1949,   17,   48,   16,  139,  101,  124,
#         390,   49,  172,   26,   20,  444,   39,   66,  549,  269,  100,
#          15,   41,   62,   92,   24,   15,   48,   19,   45,   39,   32,
#          11,   50,   10,   49,   19,   19,   24,   36,   30,   13,   21,
#          12,   18], dtype=int64))

# print(type(X_train))       # <class 'numpy.ndarray'>
# print(type(X_train[0]))     # <class 'list'>

X_train = pad_sequences(X_train, padding='pre', maxlen=1024, truncating='pre')
X_test = pad_sequences(X_test, padding='pre', maxlen=1024, truncating='pre')
# y 원핫은 하고싶으면 하고 하기 싫으면 sparse_categorical_crossentropy
print(X_train.shape)    # (8982, 100)
print(X_test.shape)     # (2246, 100)

# y_train = np.array(y_train)
# y_test = np.array(y_test)

ohe = OneHotEncoder(sparse=False)
y_train = y_train.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)
y_train = ohe.fit_transform(y_train)
y_test = ohe.transform(y_test)

# print(np.unique(X_train, return_counts=True))

print(y_train.shape)    # (8982, 46)
print(y_test.shape)     # (2246, 46)
# X_train = X_train.reshape(-1, 100, 1)

# X_train = np.asarray(X_train).astype(np.float32)
# X_test = np.asarray(X_test).astype(np.float32)
# y_train = np.asarray(X_train).astype(np.float32)
# y_test = np.asarray(X_test).astype(np.float32)

# X_train = X_train.reshape(-1, 100, 1)
# y_train = y_train.reshape()


model = Sequential()
model.add(Embedding(input_dim=2049, output_dim=19, input_length= 1024))
model.add(Conv1D(97, 9, 19, activation='swish'))
model.add(Conv1D(9, 21, 97, activation='swish'))
model.add(GRU(21, return_sequences=True))
model.add(Flatten())                                                   
model.add(Dense(16, activation='relu'))
model.add(Dense(21, activation='relu'))
model.add(Dense(46,activation='softmax'))
model.summary()





model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
es = EarlyStopping(monitor='val_loss', mode='min', patience=100, verbose=2, restore_best_weights=True)
model.fit(X_train, y_train, epochs=500, batch_size=500, verbose=2, validation_split=0.15, callbacks=[es])


results = model.evaluate(X_test, y_test)

print("로스 : ", results[0])
print("acc : ", results[1])



# 로스 :  1.6726274490356445
# acc :  0.5979518890380859

# 로스 :  1.5943092107772827
# acc :  0.613089919090271

# 로스 :  1.6859161853790283
# acc :  0.6010685563087463