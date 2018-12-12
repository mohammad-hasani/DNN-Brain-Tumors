import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.optimizers import SGD, Adam, RMSprop
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
import random
from sklearn.metrics import confusion_matrix
from tools import data_flatten


np.random.seed(2018)


class ANN(object):
    def __init__(self, X, y, testsize=.10):
        self.NB_EPOCH = 200
        self.BATCH_SIZE = 300
        self.VERBOSE = 1
        self.NB_CLASSES = 2
        self.OPTIMIZER = SGD()
        self.N_HIDDEN = 512
        self.VALIDATION_SPLIT = 0.14

        self.RESHAPED = 31488

        X = data_flatten(X)

        rnd = np.arange(X.shape[0])
        random.shuffle(rnd)

        X = X[rnd]
        y = y[rnd]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=testsize)

        X_train = X_train.astype('float32')
        X_test = X_test.astype('float32')

        X_train /= 255
        X_test /= 255

        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

        self.y_train = np_utils.to_categorical(self.y_train, self.NB_CLASSES)
        self.y_test = np_utils.to_categorical(self.y_test, self.NB_CLASSES)

    def ann(self):
        model = Sequential()
        model.add(Dense(self.N_HIDDEN, input_shape=(self.RESHAPED,)))
        model.add(Activation('relu'))
        model.add(Dense(self.N_HIDDEN, input_shape=(self.RESHAPED,)))
        model.add(Activation('relu'))
        model.add(Dense(self.NB_CLASSES))
        model.add(Activation('softmax'))
        model.summary()
        model.compile(loss='categorical_crossentropy',
                      optimizer=self.OPTIMIZER,
                      metrics=['accuracy'])
        history = model.fit(self.X_train, self.y_train,
                            batch_size=self.BATCH_SIZE, epochs=self.NB_EPOCH,
                            verbose=self.VERBOSE, validation_split=self.VALIDATION_SPLIT)
        score = model.evaluate(self.X_test, self.y_test, verbose=self.VERBOSE)
        print('Test score: ', score[0])
        print('Test accuracy: ', score[1])

        y_predict = model.predict_classes(self.X_test)
        y_test = np.argmax(self.y_test, axis=1)

        conv_matrix = confusion_matrix(y_test, y_predict)
        print(conv_matrix)

        model.save('./Models/ANN.h5')

        return model