from keras.datasets import imdb
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


(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=10000)

# print(X_train)
# print(X_train.shape, y_train.shape)    #  (25000,) (25000,)
# print(X_test.shape, y_test.shape)       # (25000,) (25000,)
# print(len(X_train[0]), len(X_test[0]))    # 218 68
# print(y_train[:20])                         # [1 0 0 1 0 0 1 0 1 0 1 0 0 0 0 0 1 1 0 1]
# print(np.unique(y_train, return_counts=True))
# (array([0, 1], dtype=int64), array([12500, 12500], dtype=int64))

# print("imdb의 최대길이 : ", max(len(i) for i in X_train))           # 2494
# print("imdb의 평균거리 : ", sum(map(len,X_train))/ len(X_train))    # 238.71364


X_train = pad_sequences(X_train, padding='pre', maxlen=200, truncating='pre')
X_test = pad_sequences(X_test, padding='pre', maxlen=200, truncating='pre')

# print(X_test.shape)     # (25000, 200)
# print(X_train.shape)    # (25000, 200)

ohe = OneHotEncoder(sparse=False)
y_train = y_train.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)
y_train = ohe.fit_transform(y_train)
y_test = ohe.transform(y_test)


# print(y_train)
# print(y_train.shape)        # (25000, 2)
# print(y_test.shape)         # (25000, 2)


model = Sequential()
model.add(Embedding(input_dim=10001, output_dim=19, input_length= 200))
model.add(LSTM(97, return_sequences=True))
model.add(GRU(9))                                                   
model.add(Dense(21, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(21, activation='relu'))
model.add(Dense(2,activation='softmax'))
model.summary()

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
es = EarlyStopping(monitor='val_loss', mode='min', patience=100, verbose=2, restore_best_weights=True)
model.fit(X_train, y_train, epochs=500, batch_size=50, verbose=2, validation_split=0.15, callbacks=[es])


results = model.evaluate(X_test, y_test)

print("로스 : ", results[0])
print("acc : ", results[1])


# epochs = 500, batch_size = 500


# 로스 :  0.3234172463417053
# acc :  0.8611199855804443

