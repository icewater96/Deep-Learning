# -*- coding: utf-8 -*-
"""
Demo MNIST with keras
"""

# From https://github.com/wxs/keras-mnist-tutorial/blob/master/MNIST%20in%20Keras.ipynb

#import matplotlib
#matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


import numpy as np
import seaborn as sns
import time
#plt.rcParams['figure.figsize'] = (7,7)

#import pydot

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.utils import np_utils
from keras import initializations
#from keras.utils.visualize_util import plot
from keras_utilities import MonitorWeightsCallback
from keras_utilities import MonitorMetricsCallback

def my_init(shape, name=None):
    return initializations.normal(shape, scale=0.01, name=name)

        
#%% Main
        
nb_classes = 10

# Load data, shuffle and split into train and test sets
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

print X_train.shape
print Y_train.shape

#plt.figure(num=2)
#for i in range(9):
#    plt.subplot(3,3,i+1)
#    plt.imshow(X_train[i], cmap='gray', interpolation='none')
#    plt.title('Class {}'.format(Y_train[i]))
    
    
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
model.add(Dense(625, input_shape=(784,), init=my_init) )
model.add(Activation('relu'))
model.add(Dropout(0.5))

#model.add(Dense(512))
#model.add(Activation('relu'))
#model.add(Dropout(0.0))

#model.add(Dense(512))
#model.add(Activation('relu'))
#model.add(Dropout(0.0))
#
#model.add(Dense(512))
#model.add(Activation('relu'))
#model.add(Dropout(0.0))

#model.add(Dense(1024))
#model.add(Activation('relu'))
#model.add(Dropout(0.4))
#
#model.add(Dense(1024))
#model.add(Activation('relu'))
#model.add(Dropout(0.4))
#
#model.add(Dense(1024))
#model.add(Activation('relu'))
#model.add(Dropout(0.4))

model.add(Dense(10, init=my_init) )
model.add(Activation('softmax'))

# Compile
model.compile(loss='categorical_crossentropy', metrics = ['accuracy'], optimizer='rmsprop')

# Callback object 
weights_callback = MonitorWeightsCallback()
metrics_callback = MonitorMetricsCallback(True, 3)

# Train 
# Hooked with callback object
model.fit(X_train, Y_train,
          batch_size = 128, nb_epoch = 10,
          verbose = 2,
          validation_data = (X_test, Y_test),
          callbacks =[weights_callback, metrics_callback])

# Collect callback results
#callback
          
# Evaluate
score = model.evaluate(X_test, Y_test, verbose = 0)          
                       
