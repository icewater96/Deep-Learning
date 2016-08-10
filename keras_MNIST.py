# -*- coding: utf-8 -*-
"""
Demo MNIST with keras
"""

# From https://github.com/wxs/keras-mnist-tutorial/blob/master/MNIST%20in%20Keras.ipynb


import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (7,7)

import pydot

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.utils import np_utils
#from keras.utils.visualize_util import plot


nb_classes = 10

# Load data, shuffle and split into train and test sets
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

print X_train.shape
print Y_train.shape

for i in range(9):
    plt.subplot(3,3,i+1)
    plt.imshow(X_train[i], cmap='gray', interpolation='none')
    plt.title('Class {}'.format(Y_train[i]))
    
    
# Format data for training
X_train = X_train.reshape(60000, 784)
X_test = X_test.reshape(10000, 784)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
print X_train.shape
print Y_train.shape

Y_train = np_utils.to_categorical(Y_train, nb_classes)
Y_test = np_utils.to_categorical(Y_test, nb_classes)

# Construct model
model = Sequential()
model.add(Dense(512, input_shape=(784,)))
model.add(Activation('relu'))
model.add(Dropout(0.4))

model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.4))

model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.4))

model.add(Dense(10))
model.add(Activation('softmax'))

# Compile
model.compile(loss='categorical_crossentropy', metrics = ['accuracy'], optimizer='adam')

# Train
model.fit(X_train, Y_train,
          batch_size = 128, nb_epoch = 20,
          verbose = 2,
          validation_data = (X_test, Y_test))
          
# Evaluate
score = model.evaluate(X_test, Y_test, verbose = 0)          
                       
