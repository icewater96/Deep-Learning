# -*- coding: utf-8 -*-
"""
Utitlities for keras

@author: JLLU
"""

# Good info at https://github.com/fchollet/keras/pull/171

import keras
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

class MonitorWeightsCallback(keras.callbacks.Callback):
    def __init__(self, nb_epoch=100):
        # Number of digits in epoch name
        self.epoch_digit_count = int(np.ceil(np.log10(nb_epoch + 0.5)))
        
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
        # weight_structure: based on model.layers[index].get_weights()
        #   mirror the structure of model.layers            
        #   model.layers is a list of layers
        #   model.layers[index].get_weights() is a list, each element is a weight array. 
        #       For Dense layer, get_weights() is a 2-element list, 0th for matrix and 1th for bias
        #  weight_structure: a list, each element is for a layer (called layer-element)
        #   each layer-element is a list, each element is a weight-array dict: key = epoch name, value = flattered weights in the epoch
        self.weight_structure_flattened = []
        self.weight_structure_original  = []
        for layer in self.model.layers:
            layer_element_flattened = []
            layer_element_original = []
            for weight_array in layer.get_weights():
                # weight_array is a numpy.ndarray
                layer_element_flattened.append( {'Epoch ' + self.generate_epoch_name(0): weight_array.reshape(np.prod(weight_array.shape))})
                layer_element_original.append ( {'Epoch ' + self.generate_epoch_name(0): weight_array})
                
            self.weight_structure_flattened.append(layer_element_flattened)
            self.weight_structure_original.append (layer_element_original )
            
            
    def on_epoch_end(self, batch, logs={}):
        self.epoch += 1
     
        # Update weight_structure
        for layer_index, layer in enumerate(self.model.layers):
            for weight_array_index, weight_array in enumerate(layer.get_weights()):
                # weight_array is a numpy.ndarray
                self.weight_structure_flattened[layer_index][weight_array_index]['Epoch ' + self.generate_epoch_name(self.epoch)] = \
                    weight_array.reshape(np.prod(weight_array.shape))
                self.weight_structure_original[layer_index][weight_array_index]['Epoch ' + self.generate_epoch_name(self.epoch)] = \
                    weight_array
                        
    def display_one_flattened_weight_array(self, layer_index=0, param_index=0, bins=None):
        # Call this function only after taininng is done
        # layer_index and param_index start at 0. param_index = weight_array index
        # bins: results of np.linspace()
    
        if bins is None:
            bins = np.linspace(-1, 1, 50)
            
        plt.figure(figsize=(15, 10))
        
        weight_array_dict = self.weight_structure_flattened[layer_index][param_index]
        epoch_name_list = weight_array_dict.keys()
        epoch_name_list.sort()
        num_subplot = len(epoch_name_list) + 1  # +1 is for text subplot
        num_row = np.round(np.sqrt(num_subplot))
        num_col = np.ceil( float(num_subplot) / num_row )

        for subplot_index, epoch_name in enumerate(epoch_name_list):
            temp_ax = plt.subplot(num_row, num_col, subplot_index + 1)
            plt.hist(weight_array_dict[epoch_name], bins)
            #plt.title(epoch_name)
            temp_ax.set_xticklabels([])
            temp_ax.set_yticklabels([])     
            
            temp_xlim = temp_ax.get_xlim()
            temp_ylim = temp_ax.get_ylim()
            temp_ax.text(temp_xlim[0] + 0.05 * (temp_xlim[1] - temp_xlim[0] ), 
                         temp_ylim[0] + 0.9 * (temp_ylim[1] - temp_ylim[0] ),
                         epoch_name )
            
        # Text subplot
        temp_ax = plt.subplot(num_row, num_col, num_subplot)            
        temp_ax.plot([0, 1], [0, 1], '.')
        temp_ax.text( 0.2, 0.5, 'Layer = ' + str(layer_index) + '\n' +  
                                'Param = ' + str(param_index) + '\n' +
                                'Min bin = ' + str(min(bins)) + '\n' + 
                                'Max bin = ' + str(max(bins)) + '\n',
                     multialignment='left')
        temp_ax.set_xticklabels([])
        temp_ax.set_yticklabels([])  
        
        plt.tight_layout()
            
        #sns.violinplot( pd.DataFrame(self.weight_structure_flattened[layer_index][param_index]), orient='h' )
        
        
#    def display_all_flattened_weight_arrays(self):
        

class MonitorMetricsCallback(keras.callbacks.Callback):
    def __init__(self, live_display=True, display_interval=1, figsize=(10, 7)):
        # Initialize the figure and axis
        self.fig = plt.figure(figsize=figsize)
        
        # Display as the training proceeds
        self.live_display = live_display
        
        # display_interval: how frequently display() is executed. 
        self.display_interval = display_interval

        # epoch = 1 is the 1st epoch    
        # epoch = 0 is the intialization
        self.epoch = 0
        
        # Internal state
        self.displayed_in_this_epoch = False
    
    def on_train_begin(self, logs={}):
        # Metrics
        self.metrics = {}
        for metrics in self.model.metrics_names:
            self.metrics[metrics] = []
            self.metrics['val_'+metrics] = []
            
    def on_epoch_begin(self, batch, logs={}):
        self.displayed_in_this_epoch = False
        
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
                self.displayed_in_this_epoch = True                
            
            
    def on_train_end(self, logs={}):
        if not(self.displayed_in_this_epoch):
            self.display()
            
    def display(self):
        # For display
        plt.figure(self.fig.number)
        plt.clf()

        num_subplot = len(self.model.metrics_names)
        for subplot_index in range(num_subplot):
            plt.subplot(num_subplot, 1, subplot_index+1)
            
            train_metrics_name = self.model.metrics_names[subplot_index]                
            test_metrics_name  = 'val_' + train_metrics_name
        
            if len(self.metrics[train_metrics_name]) <= 30:
                plt.plot(range(1, len(self.metrics[train_metrics_name]) + 1), self.metrics[train_metrics_name], 'b-o')
                plt.plot(range(1, len(self.metrics[test_metrics_name ]) + 1), self.metrics[test_metrics_name ], 'r-o')
            else:
                plt.plot(range(1, len(self.metrics[train_metrics_name]) + 1), self.metrics[train_metrics_name], 'b-')
                plt.plot(range(1, len(self.metrics[test_metrics_name ]) + 1), self.metrics[test_metrics_name ], 'r-')
            legend_list = [train_metrics_name, test_metrics_name]

            plt.legend(legend_list, loc='best')    
            if subplot_index == num_subplot - 1:
                plt.xlabel('Epoch')
            if subplot_index == 0:
                plt.title('Epoch = ' + str(self.epoch))

        # Make sure the monitoring plot
        #plt.show()
        plt.pause(1)        
        

