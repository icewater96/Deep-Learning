# -*- coding: utf-8 -*-
"""
Demo MNIST with keras with MLP
"""

# Baseline is from https://github.com/fchollet/keras/blob/master/examples/mnist_mlp.py

#import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.utils import np_utils
from keras_utilities import MonitorWeightsCallback
from keras_utilities import MonitorMetricsCallback

import numpy as np
np.random.seed(1337)  # for reproducibility

batch_size = 128
nb_classes = 10
nb_epoch = 20


# Load data, shuffle and split into train and test sets
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

print X_train.shape
print Y_train.shape


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
model.add(Dropout(0.2))

model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.2))

model.add(Dense(10))
model.add(Activation('softmax'))

model.summary()

# Compile
model.compile(loss='categorical_crossentropy', metrics = ['accuracy'], optimizer='rmsprop')

# Callback object 
weights_callback = MonitorWeightsCallback(nb_epoch)
metrics_callback = MonitorMetricsCallback(True, 3)

# Train 
# Hooked with callback object
history = model.fit(X_train, Y_train,
                    batch_size=batch_size,
                    nb_epoch=nb_epoch,
                    verbose = 2,
                    validation_data = (X_test, Y_test),
                    callbacks =[weights_callback, metrics_callback])

# Collect callback results
#callback
          
# Evaluate
#score = model.evaluate(X_test, Y_test, verbose = 0)          
                       