from keras import backend as K
from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers import Dropout
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils
from keras.optimizers import SGD, RMSprop, Adam
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import random
from keras.utils import to_categorical
from sklearn.metrics import confusion_matrix


class DCNN(object):
    def __init__(self, X, Y, testsize=.10):
        self.NB_EPOCH = 20
        self.BATCH_SIZE = 300
        self.VERBOSE = 1
        self.NB_CLASSES = 2
        self.OPTIMIZER = Adam()
        self.N_HIDDEN = 64
        self.VALIDATION_SPLIT = 0.13

        self.IMG_ROWS = 256
        self.IMG_COLS = 123
        self.IMG_CHANNELS = 1

        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=testsize)

        X_train = X_train.astype('float32')
        X_test = X_test.astype('float32')

        X_train /= 255
        X_test /= 255

        self.X_train = X_train
        self.Y_train = y_train
        self.X_test = X_test
        self.Y_test = y_test

        self.Y_train = np_utils.to_categorical(self.Y_train, self.NB_CLASSES)
        self.Y_test = np_utils.to_categorical(self.Y_test, self.NB_CLASSES)


    def dcnn(self):
        print(self.IMG_ROWS)
        print(self.IMG_COLS)
        print(self.IMG_CHANNELS)
        print(self.X_train.shape)
        self.X_train = self.X_train.reshape(self.X_train.shape[0], self.IMG_ROWS, self.IMG_COLS, 1)
        print(self.X_train.shape)
        model = Sequential()
        model.add(Conv2D(self.N_HIDDEN, (3, 3), padding='same', input_shape=(self.IMG_ROWS, self.IMG_COLS, 1)))
        model.add(Activation('relu'))
        model.add(Conv2D(self.N_HIDDEN, (3, 3), padding='same'))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Conv2D(self.N_HIDDEN, (3, 3), padding='same'))
        model.add(Activation('relu'))
        model.add(Conv2D(self.N_HIDDEN, 3, 3))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(512))
        model.add(Activation('relu'))
        model.add((Dropout(0.5)))
        model.add(Dense(2))
        model.add(Activation('softmax'))


        model.compile(optimizer=self.OPTIMIZER, loss='mse', metrics=['accuracy'])
        model.summary()

        # history = model.fit(self.X_train, self.Y_train,
        #                     batch_size=self.BATCH_SIZE, epochs=self.NB_EPOCH,
        #                     verbose=self.VERBOSE, validation_split=self.VALIDATION_SPLIT)

        history = model.fit(self.X_train, self.Y_train,
                            epochs=self.NB_EPOCH,
                            verbose=self.VERBOSE)

        y_predict = model.predict_classes(self.Y_test)
        print(y_predict)
        # y_test = np.argmax(self.y_test, axis=1)

        conv_matrix = confusion_matrix(self.Y_test, y_predict)
        print(conv_matrix)
        model.save('./Models/DCNN.h5')
        return model













