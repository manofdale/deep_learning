from __future__ import absolute_import
from __future__ import print_function

import numpy as np

from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.utils import np_utils

from util import misc

import random


def add_random_convolution(model, nb_filters, nb_conv, nb_pool, dropout_rate, activation_func, layer_limit=6, ):
    for i in range(random.randint(0, layer_limit - 1)):  # no more than limit
        model.add(Convolution2D(nb_filters, nb_conv, nb_conv))
        model.add(Activation(activation_func))
    model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
    model.add(Dropout(dropout_rate))


def add_random_dense(model, hidden_layer_size, activation_function, dropout_rate, layer_limit=3):
    for i in range(random.randint(0, layer_limit - 1)):
        model.add(Dense(hidden_layer_size))
        model.add(Activation(activation_function))
        model.add(Dropout(dropout_rate))


def random_construct_cnn(model, img_rows=28, img_cols=28, cnn_limit=4, nb_filters=42, nb_pool=2,
                         nb_conv=3, nb_classes=47, dropout_limit=0.5, hidden_layer_limit=1024,
                         border_mode='same', optimizer='adadelta',
                         loss_function='categorical_crossentropy',
                         cnn_activation_functions=['relu'], dense_activation_function='relu', final_activation='softmax'
                         ):
    model.add(Convolution2D(nb_filters, nb_conv, nb_conv,
                            border_mode=border_mode,
                            input_shape=(1, img_rows, img_cols)))
    hard_limit = min(img_cols, img_rows)
    for i in range(cnn_limit):
        nb_conv = 2 + random.random(nb_conv)
        nb_pool = 2 + random.random(nb_pool)
        hard_limit //= nb_pool
        if hard_limit < 4:
            break
        activation_func = cnn_activation_functions[random.randint(0, len(cnn_activation_functions) - 1)]
        dropout_rate = random.random() * dropout_limit * 0.5
        layer_limit = random.randint(2, 6)
        add_random_convolution(model, nb_filters, nb_conv, nb_pool, dropout_rate, activation_func, layer_limit)
    model.add(Flatten())
    hidden_layer_size = random.randint(hard_limit ** 2, hard_limit ** 4)
    dropout_rate = random.random() * dropout_limit
    layer_limit = random.randint(2, 4)
    add_random_dense(model, hidden_layer_size, dense_activation_function, dropout_rate, layer_limit=layer_limit)
    model.add(Dense(nb_classes))
    model.add(Activation(final_activation))

    model.compile(loss=loss_function, optimizer=optimizer)
