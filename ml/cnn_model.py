"""
Created on Nov 19, 2015

@author: agp
"""
from __future__ import absolute_import
from __future__ import print_function

import numpy as np

from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.utils import np_utils

from util import misc

np.random.seed(1337)  # for reproducibility


# takes too much time to train
def prepare_model(model, img_rows=28, img_cols=28, nb_filters=42, nb_pool=2,
                  nb_conv=3, nb_classes=10, dropout_rates=[0.1, 0.2, 0.5],
                  border_mode='same', optimizer='adadelta',
                  loss_function='categorical_crossentropy',
                  activation_functions=['relu', 'relu', 'relu', 'relu', 'softmax'], hidden_layers=[128]):
    """setup architecture and prepare modelfor training
    @param nb_filters: number of convolutional filters to use
    @param nb_pool: size of pooling area for max pooling
    @param nb_conv: convolution kernel size
    """
    model.add(Convolution2D(nb_filters, nb_conv, nb_conv,
                            border_mode=border_mode,
                            input_shape=(1, img_rows, img_cols)))
    model.add(Activation(activation_functions[0]))
    model.add(Convolution2D(nb_filters, nb_conv, nb_conv))
    model.add(Activation(activation_functions[1]))
    model.add(Convolution2D(nb_filters, nb_conv, nb_conv))
    model.add(Activation(activation_functions[1]))
    model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
    model.add(Dropout(dropout_rates[0]))
    model.add(Convolution2D(nb_filters, nb_conv, nb_conv))
    model.add(Activation(activation_functions[1]))
    model.add(Convolution2D(nb_filters, nb_conv, nb_conv))
    model.add(Activation(activation_functions[1]))
    model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
    model.add(Dropout(dropout_rates[0]))
    model.add(Convolution2D(nb_filters, nb_conv, nb_conv))
    model.add(Activation(activation_functions[1]))
    model.add(Convolution2D(nb_classes, nb_conv, nb_conv))
    model.add(Activation(activation_functions[2]))
    model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
    model.add(Dropout(dropout_rates[1]))
    model.add(Flatten())
    model.add(Dense(hidden_layers[0]))
    model.add(Activation(activation_functions[3]))
    model.add(Dropout(dropout_rates[2]))
    model.add(Dense(nb_classes))
    model.add(Activation(activation_functions[4]))

    model.compile(loss=loss_function, optimizer=optimizer)


def prepare_rcnn(model, img_rows=28, img_cols=28, nb_filters=42, nb_pool=2,
                 nb_conv=3, nb_classes=10, border_mode='same', ):
    '''TODO replace segmentation heuristics with a trained segmenter'''
    pass


def prepare_model5(model, img_rows=28, img_cols=28, nb_filters=42, nb_pool=2,
                   nb_conv=3, nb_classes=10, dropout_rates=[0.1, 0.2, 0.5],
                   border_mode='same', optimizer='adadelta',
                   loss_function='categorical_crossentropy',
                   activation_functions=['relu', 'relu', 'relu', 'relu', 'softmax'], hidden_layers=[128]):
    """setup architecture and prepare modelfor training
    @param nb_filters: number of convolutional filters to use
    @param nb_pool: size of pooling area for max pooling
    @param nb_conv: convolution kernel size
    """
    model.add(Convolution2D(nb_filters, nb_conv, nb_conv,
                            border_mode=border_mode,
                            input_shape=(1, img_rows, img_cols)))
    model.add(Activation(activation_functions[0]))
    model.add(Convolution2D(nb_filters, nb_conv, nb_conv))
    model.add(Activation(activation_functions[1]))
    model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
    model.add(Dropout(dropout_rates[0]))
    model.add(Convolution2D(nb_filters, nb_conv, nb_conv))
    model.add(Activation(activation_functions[1]))
    model.add(Convolution2D(nb_classes, nb_conv, nb_conv))
    model.add(Activation(activation_functions[2]))
    model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
    model.add(Dropout(dropout_rates[1] / 1.5))
    model.add(Flatten())
    model.add(Dense(hidden_layers[0]))
    model.add(Activation(activation_functions[3]))
    model.add(Dropout(dropout_rates[2]))
    model.add(Dense(hidden_layers[0] // 4))
    model.add(Activation(activation_functions[3]))
    model.add(Dropout(dropout_rates[2]))
    model.add(Dense(nb_classes))
    model.add(Activation(activation_functions[4]))

    model.compile(loss=loss_function, optimizer=optimizer)


def prepare_model6(model, img_rows=28, img_cols=28, nb_filters=42, nb_pool=2,
                   nb_conv=3, nb_classes=10, dropout_rates=[0.1, 0.2, 0.5],
                   border_mode='same', optimizer='adadelta',
                   loss_function='categorical_crossentropy',
                   activation_functions=['relu', 'relu', 'relu', 'relu', 'softmax'], hidden_layers=[128]):
    """setup architecture and prepare modelfor training
    @param nb_filters: number of convolutional filters to use
    @param nb_pool: size of pooling area for max pooling
    @param nb_conv: convolution kernel size
    """
    model.add(Convolution2D(nb_filters, nb_conv, nb_conv,
                            border_mode=border_mode,
                            input_shape=(1, img_rows, img_cols)))
    model.add(Activation(activation_functions[0]))
    model.add(Convolution2D(nb_filters, nb_conv, nb_conv))
    model.add(Activation(activation_functions[1]))
    model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
    model.add(Dropout(dropout_rates[0]))
    model.add(Convolution2D(nb_filters, nb_conv, nb_conv))
    model.add(Activation(activation_functions[1]))
    model.add(Convolution2D(nb_classes, nb_conv, nb_conv))
    model.add(Activation(activation_functions[2]))
    model.add(Convolution2D(nb_classes, nb_conv, nb_conv))
    model.add(Activation(activation_functions[2]))
    model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
    model.add(Dropout(dropout_rates[1]))
    model.add(Flatten())
    model.add(Dense(hidden_layers[0]))
    model.add(Activation(activation_functions[3]))
    model.add(Dropout(dropout_rates[2]))
    model.add(Dense(nb_classes))
    model.add(Activation(activation_functions[4]))

    model.compile(loss=loss_function, optimizer=optimizer)


def prepare_model4(model, img_rows=28, img_cols=28, nb_filters=42, nb_pool=2,
                   nb_conv=3, nb_classes=10, dropout_rates=[0.1, 0.2, 0.5],
                   border_mode='same', optimizer='adadelta',
                   loss_function='categorical_crossentropy',
                   activation_functions=['relu', 'relu', 'relu', 'relu', 'softmax'], hidden_layers=[128]):
    """setup architecture and prepare modelfor training
    @param nb_filters: number of convolutional filters to use
    @param nb_pool: size of pooling area for max pooling
    @param nb_conv: convolution kernel size
    """
    model.add(Convolution2D(nb_filters, nb_conv, nb_conv,
                            border_mode=border_mode,
                            input_shape=(1, img_rows, img_cols)))
    model.add(Activation(activation_functions[0]))
    model.add(Convolution2D(nb_filters, nb_conv, nb_conv))
    model.add(Activation(activation_functions[1]))
    model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
    model.add(Dropout(dropout_rates[0]))
    model.add(Convolution2D(nb_filters, nb_conv, nb_conv))
    model.add(Activation(activation_functions[1]))
    model.add(Convolution2D(nb_classes, nb_conv, nb_conv))
    model.add(Activation(activation_functions[2]))
    model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
    model.add(Dropout(dropout_rates[1] / 1.5))
    model.add(Flatten())
    model.add(Dense(hidden_layers[0]))
    model.add(Activation(activation_functions[3]))
    model.add(Dropout(dropout_rates[2]))
    model.add(Dense(hidden_layers[0] // 2))
    model.add(Activation(activation_functions[3]))
    model.add(Dropout(dropout_rates[2] / 1.5))
    model.add(Dense(hidden_layers[0] // 4))
    model.add(Activation(activation_functions[3]))
    model.add(Dropout(dropout_rates[2] / 2.5))
    model.add(Dense(nb_classes))
    model.add(Activation(activation_functions[4]))

    model.compile(loss=loss_function, optimizer=optimizer)


def prepare_model2(model, img_rows=28, img_cols=28, nb_filters=42, nb_pool=2,
                   nb_conv=3, nb_classes=10, dropout_rates=[0.1, 0.2, 0.5],
                   border_mode='same', optimizer='adadelta',
                   loss_function='categorical_crossentropy',
                   activation_functions=['relu', 'relu', 'relu', 'relu', 'softmax'], hidden_layers=[128]):
    """setup architecture and prepare modelfor training
    @param nb_filters: number of convolutional filters to use
    @param nb_pool: size of pooling area for max pooling
    @param nb_conv: convolution kernel size
    """
    model.add(Convolution2D(nb_filters, nb_conv, nb_conv,
                            border_mode=border_mode,
                            input_shape=(1, img_rows, img_cols)))
    model.add(Activation(activation_functions[0]))
    model.add(Convolution2D(nb_filters, nb_conv, nb_conv))
    model.add(Activation(activation_functions[1]))
    model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
    model.add(Dropout(dropout_rates[0]))
    model.add(Convolution2D(nb_filters, nb_conv, nb_conv))
    model.add(Activation(activation_functions[1]))
    model.add(Convolution2D(nb_classes, nb_conv, nb_conv))
    model.add(Activation(activation_functions[2]))
    model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
    model.add(Dropout(dropout_rates[1]))
    model.add(Flatten())
    model.add(Dense(hidden_layers[0]))
    model.add(Activation(activation_functions[3]))
    model.add(Dropout(dropout_rates[2]))
    model.add(Dense(hidden_layers[0] // 2))
    model.add(Activation(activation_functions[3]))
    model.add(Dropout(dropout_rates[2] / 1.5))
    model.add(Dense(nb_classes))
    model.add(Activation(activation_functions[4]))

    model.compile(loss=loss_function, optimizer=optimizer)


def prepare_model3(model, img_rows=28, img_cols=28, nb_filters=42, nb_pool=2,
                   nb_conv=3, nb_classes=10, dropout_rates=[0.1, 0.2, 0.5],
                   border_mode='same', optimizer='adadelta',
                   loss_function='categorical_crossentropy',
                   activation_functions=['relu', 'relu', 'relu', 'relu', 'softmax'], hidden_layers=[128]):
    """setup architecture and prepare modelfor training
    @param nb_filters: number of convolutional filters to use
    @param nb_pool: size of pooling area for max pooling
    @param nb_conv: convolution kernel size
    """
    model.add(Convolution2D(nb_filters, nb_conv, nb_conv,
                            border_mode=border_mode,
                            input_shape=(1, img_rows, img_cols)))
    model.add(Activation(activation_functions[0]))
    model.add(Convolution2D(nb_filters, nb_conv, nb_conv))
    model.add(Activation(activation_functions[1]))
    model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
    model.add(Dropout(dropout_rates[0]))
    model.add(Convolution2D(nb_filters, nb_conv, nb_conv))
    model.add(Activation(activation_functions[1]))
    model.add(Convolution2D(nb_classes, nb_conv, nb_conv))
    model.add(Activation(activation_functions[2]))
    model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
    model.add(Dropout(dropout_rates[1]))
    model.add(Flatten())
    model.add(Dense(hidden_layers[0]))
    model.add(Activation(activation_functions[3]))
    model.add(Dropout(dropout_rates[2]))
    model.add(Dense(nb_classes))
    model.add(Activation(activation_functions[4]))

    model.compile(loss=loss_function, optimizer=optimizer)


def reshape_input(x_train, img_rows=28, img_cols=28):
    """convert gray values (0:255) to floating points (0:1) and change the shape"""
    x_train = np.array(x_train).astype("float32") / 255.0
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    return x_train


def reshape_int_output(y_train, nb_classes=10):
    """ convert class vectors to binary class matrices"""
    return np_utils.to_categorical(np.array(y_train), nb_classes)


def reshape_str_output(y_train, nb_classes=62):
    mapper = misc.Mapper()
    misc.init_mapper(mapper)
    return reshape_int_output(misc.map_from_mapper(y_train, mapper, left=True), nb_classes)
