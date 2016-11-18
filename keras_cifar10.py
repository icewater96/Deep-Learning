# -*- coding: utf-8 -*-
"""
https://github.com/fchollet/keras/blob/master/examples/cifar10_cnn.py

@author: JLLU
"""



from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
from keras.utils import np_utils
from keras_utilities import MonitorWeightsCallback
from keras_utilities import MonitorMetricsCallback

batch_size = 100
nb_classes = 10
nb_epoch = 30
data_augmentation = False

# Input image dimensions
img_rows, img_cols = 32, 32
# CIFAR10 images are RGB
img_channels = 3

# Shuffle and split
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
print 'X_train shape: ' + str(X_train.shape)
print str(X_train.shape[0]) + ' train samples'
print str(X_test.shape[0]) +  ' test samples'

# Convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test  = np_utils.to_categorical(y_test , nb_classes)

# Model
model = Sequential()

model.add(Convolution2D(2048, 3, 3, border_mode = 'same',
                        input_shape=(img_channels, img_rows, img_cols)))
model.add(Activation('relu'))                        
model.add(Convolution2D(32, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Convolution2D(1024, 3, 3, border_mode='same'))
model.add(Activation('relu'))
model.add(Convolution2D(64, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Convolution2D(512, 3, 3, border_mode='same'))
model.add(Activation('relu'))
model.add(Convolution2D(64, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))

model.summary()

# Train the model using SGB + momentum
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])
              
# Callback object 
weights_callback = MonitorWeightsCallback(nb_epoch)
metrics_callback = MonitorMetricsCallback(True, 3)


X_train = X_train.astype('float32')
X_test  = X_test.astype ('float32')
X_train /= 255
X_test  /= 255

# Train
if not data_augmentation:
    print 'Not using data augmentaion.'
    model.fit(X_train, Y_train,
              batch_size=batch_size,
              nb_epoch=nb_epoch,
              validation_data=(X_test, Y_test),
              shuffle=True,
              callbacks =[weights_callback, metrics_callback])