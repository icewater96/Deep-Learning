# -*- coding: utf-8 -*-
"""
Created on Wed Aug 24 09:44:30 2016

@author: JLLU
"""

# Basesline https://github.com/fchollet/keras/blob/master/examples/mnist_cnn.py

import numpy as np
np.random.seed(1337)  # for reproducibility

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils

from keras_utilities import MonitorWeightsCallback
from keras_utilities import MonitorMetricsCallback

batch_size = 128
nb_classes = 10
nb_epoch = 30

# input image dimensions
img_rows, img_cols = 28, 28
# number of convolutional filters to use
nb_filters = 256  # Original one is 32
# size of pooling area for max pooling
nb_pool = 2
# convolution kernel size
kernel_size = (3, 3)

# the data, shuffled and split between train and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
print X_train.shape
print y_train.shape

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

model = Sequential()

model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1],
                        border_mode='valid',
                        input_shape=(1, img_rows, img_cols)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
model.add(Dropout(0.25))

model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1], border_mode='valid'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
model.add(Dropout(0.25))

model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1], border_mode='valid'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
model.add(Dropout(0.25))

model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1], border_mode='valid'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
model.add(Dropout(0.25))


model.add(Flatten())
model.add(Dense(256))   # Original 128
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(nb_classes))
model.add(Activation('softmax'))


# Callback object 
weights_callback = MonitorWeightsCallback(nb_epoch)
metrics_callback = MonitorMetricsCallback(True, 1)

model.compile(loss='categorical_crossentropy',
              optimizer='adadelta',
              metrics=['accuracy'])

model.fit(X_train, Y_train, 
          batch_size=batch_size, 
          nb_epoch=nb_epoch,
          verbose=1, 
          validation_data=(X_test, Y_test),
          callbacks =[weights_callback, metrics_callback])
          
#score = model.evaluate(X_test, Y_test, verbose=0)
#print('Test score:', score[0])
#print('Test accuracy:', score[1])