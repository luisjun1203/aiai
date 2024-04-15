import numpy as np
from sklearn.datasets import load_breast_cancer
from keras.models import Sequential, Model
from keras.layers import Dense, Conv2D, Flatten, MaxPool2D, Dropout, Input
from sklearn.model_selection import RandomizedSearchCV, train_test_split
X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42, stratify=y)
print(X_train.shape)
print(y_train.shape)
def build_model(drop=.5, input_shape=(30,), opti='adam', lr=0.001, node1=64, node2=32, node3=16, activation='relu'):
    inp = Input(input_shape)
    l1 = Dense(node1, activation=activation)(inp)
    l1 = Dropout(drop)(l1)
    l2 = Dense(node2, activation=activation)(l1)
    l2 = Dropout(drop)(l2)
    l3 = Dense(node3, activation=activation)(l2)
    l3 = Dropout(drop)(l3)
    outp = Dense(1, activation='sigmoid')(l2)
    model = Model(inputs=inp, outputs=outp)
    model.compile(optimizer=opti, loss='binary_crossentropy', metrics=['acc'])
    
    return model

def create_hyperparameter():
    batchs = [100, 200, 300, 400, 500]
    optimizers = ['adam', 'rmsprop', 'adadelta']
    dropouts = [0.2, 0.3, 0.4, 0.5]
    activations = ['relu', 'elu', 'swish', 'linear']
    node1 = [64, 32, 16]
    node2 = [64, 32, 16]
    node3 = [64, 32, 16]
    return {'batch_size' : batchs,
            'opti' : optimizers,
            'drop' : dropouts,
            'activation' : activations,
            'node1' : node1,
            'node2' : node2,
            'node3' : node3,
            }
    
hyperparameters = create_hyperparameter()
from keras.wrappers.scikit_learn import KerasClassifier
keras_model = KerasClassifier(build_fn=build_model, verbose=1)
model = RandomizedSearchCV(keras_model, hyperparameters, cv=3, n_iter=5, random_state=42, verbose=1)
from keras.wrappers.scikit_learn import KerasClassifier
from keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor='loss', patience=5, mode='min', restore_best_weights=True)
mcp = ModelCheckpoint('./abcd.hdf5', monitor='loss', mode='min')
model.fit(X_train, y_train, epochs=50, callbacks=[es, mcp])

print('model.best_params_ : ', model.best_params_)
print('model.best_estimator_ : ', model.best_estimator_)
print('model.best_score_ : ', model.best_score_)
print('model.score : ', model.score(X_test, y_test))
