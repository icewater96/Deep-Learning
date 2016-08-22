# -*- coding: utf-8 -*-
"""
Utitlities for keras

@author: JLLU
"""

# Good info at https://github.com/fchollet/keras/pull/171


import keras
import numpy as np
import matplotlib.pyplot as plt


class MonitorWeightsCallback(keras.callbacks.Callback):
    def __init__(self, nb_epoch=100, flatten_array=True):
        # Number of digits in epoch name
        self.epoch_digit_count = int(np.ceil(np.log10(nb_epoch + 0.5)))
        
        # Switch to flatten weight arrays
        self.flatten_array = flatten_array
        
        self.epoch = 0
        
    def generate_epoch_name(self, epoch):
        # Stardardize display of epoch number
        # epoch: the number of epoch. integer
        base = '0' * self.epoch_digit_count
        epoch_string = str(epoch)
        
        if len(epoch_string) >= len(base):
            return_value = epoch_string
        else:
            return_value = []
            delta = len(base) - len(epoch_string)
            for i in range(len(base)):
                if i < delta:
                    return_value.append(base[i])
                else:
                    return_value.append(epoch_string[i-delta])
                    
            return_value = ''.join(return_value)
        return return_value                
        
        
    def on_train_begin(self, logs={}):
      
#        # Old method based on model.get_weights()    
#        # weight_structure:
#        #   each item is a dict for a weight array. E.g. Input * W + B = Output, W and B are two weight arrays.
#        #   each weight-array dict: key = epoch name, value = flattened weights for a given epoch
#        #   get_weights() => weight_array_list
#        #   On each epoch end, need to update weight_structure with the new weight_array_list
#        self.weight_structure = []
#        weight_array_list = self.model.get_weights()
#        for weight_array in weight_array_list:
#            self.weight_structure.append( {'Epoch 0': weight_array.reshape(np.prod(weight_array.shape))} )
            
        # weight_structure: based on model.layers[index].get_weights()
        #   mirror the structure of model.layers            
        #   model.layers is a list of layers
        #   model.layers[index].get_weights() is a list, each element is a weight array. 
        #       For Dense layer, get_weights() is a 2-element list, 0th for matrix and 1th for bias
        #  weight_structure: a list, each element is for a layer (called layer-element)
        #   each layer-element is a list, each element is a weight-array dict: key = epoch name, value = flattered weights in the epoch
        self.weight_structure = []
        for layer in self.model.layers:
            layer_element = []
            for weight_array in layer.get_weights():
                # weight_array is a numpy.ndarray
                if self.flatten_array:
                    layer_element.append( {'Epoch ' + self.generate_epoch_name(0): weight_array.reshape(np.prod(weight_array.shape))})
                else:
                    layer_element.append( {'Epoch ' + self.generate_epoch_name(0): weight_array})
                
            self.weight_structure.append(layer_element)
            
            
    def on_epoch_end(self, batch, logs={}):
        self.epoch += 1
        
#        # Old method based on model.get_weights()
#        # Update weights history
#        weight_array_list = self.model.get_weights()
#        for index, weight_array in enumerate(weight_array_list):
#            self.weight_structure[index]['Epoch '+str(self.epoch)] = weight_array.reshape(np.prod(weight_array.shape))
#        
        # Update weight_structure
        for layer_index, layer in enumerate(self.model.layers):
            for weight_array_index, weight_array in enumerate(layer.get_weights()):
                # weight_array is a numpy.ndarray
                if self.flatten_array:
                    self.weight_structure[layer_index][weight_array_index]['Epoch ' + self.generate_epoch_name(self.epoch)] = \
                        weight_array.reshape(np.prod(weight_array.shape))
                else:
                    self.weight_structure[layer_index][weight_array_index]['Epoch ' + self.generate_epoch_name(self.epoch)] = \
                        weight_array
        
#    def display_one_weight_


class MonitorMetricsCallback(keras.callbacks.Callback):
    def __init__(self, live_display=True, display_interval=1, figsize=(10, 7)):
        # Initialize the figure and axis
        self.fig = plt.figure(figsize=figsize)
        
        # Display as the training proceeds
        self.live_display = live_display
        
        # display_interval: how frequently display() is executed. 
        self.display_interval = display_interval

        # Start at 1        
        self.epoch = 0
    
    def on_train_begin(self, logs={}):
        # Metrics
        self.metrics = {}
        for metrics in self.model.metrics_names:
            self.metrics[metrics] = []
            self.metrics['val_'+metrics] = []
        
    def on_epoch_end(self, batch, logs={}):
        # Update metrics history
        # Data structure of logs:
        #    {'acc': 0.18648333333333333, 'loss': 2.1393585069020591, 'val_acc': 0.3634, 'val_loss': 1.8471685892105103}
        # model.metrics_names = ['loss', 'acc']
        self.epoch += 1
        for metrics_name, value in logs.items():
            self.metrics[metrics_name].append(value)

        if self.live_display:
            if ((self.epoch % self.display_interval) == 0) or (self.epoch == 1):
                self.display()
            
            
    def on_train_end(self, logs={}):
        if not(self.live_display) or (self.display_interval != 1):
            self.display()

            
    def display(self):
        # For display
        plt.figure(self.fig.number)
        #ax = metrics_figure.get_axes()[0]  # Assume just 1 axes
        plt.clf()

        num_subplot = len(self.model.metrics_names)
        for subplot_index in range(num_subplot):
            plt.subplot(num_subplot, 1, subplot_index+1)
            
            train_metrics_name = self.model.metrics_names[subplot_index]                
            test_metrics_name  = 'val_' + train_metrics_name
        
            plt.plot(self.metrics[train_metrics_name], 'b-o')
            plt.plot(self.metrics[test_metrics_name ], 'r-o')
            legend_list = [train_metrics_name, test_metrics_name]

            plt.legend(legend_list, loc='best')    
            if subplot_index == num_subplot - 1:
                plt.xlabel('Epoch')
            if subplot_index == 0:
                plt.title('Epoch = ' + str(self.epoch))

        # Make sure the monitoring plot
        #plt.show()
        plt.pause(1)        
        

