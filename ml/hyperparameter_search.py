from __future__ import absolute_import
from __future__ import print_function

import numpy as np

from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.models import Sequential
from keras.utils import np_utils

from util import misc

import random


def add_convolution(model, nb_filters, nb_conv, nb_pool, dropout_rate, activation_func, r):
    for i in range(r):  # no more than limit
        model.add(Convolution2D(nb_filters, nb_conv, nb_conv))
        model.add(Activation(activation_func))
    model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
    model.add(Dropout(dropout_rate))


def add_dense(model, hidden_layer_size, activation_function, dropout_rate, r):
    for i in range(r):
        model.add(Dense(hidden_layer_size))
        model.add(Activation(activation_function))
        model.add(Dropout(dropout_rate))


def random_cnn_config(img_rows=28, img_cols=28, dense_limit=3, cnn_limit=4, nb_filters=42,
                         nb_pool=2,
                         nb_conv=3, nb_classes=47, cnn_dropout_limit=0.25, dropout_limit=0.75, hidden_layer_limit=1024,
                         border_modes=['same'], optimizers=['adadelta'],
                         loss_functions=['categorical_crossentropy'],
                         cnn_activation_functions=['relu'], dense_activation_functions=['relu'],
                         final_activation='softmax',
                         ):
    border_mode = border_modes[random.randint(0, len(border_modes) - 1)]
    nb_conv = 2 + random.randint(0, nb_conv)
    nb_filter = 10 + random.randint(0, nb_filters)
    activation_func = cnn_activation_functions[random.randint(0, len(cnn_activation_functions) - 1)]
    config = {"nb_conv": [nb_conv], "border_mode": border_mode, "img_rows": img_rows, "img_cols": img_cols,
              "nb_classes": nb_classes,
              "nb_pool": [], "nb_filter": [nb_filter], "dropout": [],
              "activation": [activation_func], "dense_layer_size": [], "nb_repeat": []}
    hard_limit = min(img_cols, img_rows)
    cnn_layer_n = 0
    for i in range(cnn_limit):
        cnn_layer_n += 1
        conv = 2 + random.randint(0, nb_conv)
        pool = 2 + random.randint(0, nb_pool)
        hard_limit //= nb_pool
        if hard_limit < 4:
            break
        activation_func = cnn_activation_functions[random.randint(0, len(cnn_activation_functions) - 1)]
        dropout_rate = random.random() * cnn_dropout_limit
        nb_filter = 10 + random.randint(0, nb_filters)
        layer_limit = random.randint(2, 6)
        r = 1 + random.randint(0, layer_limit - 1)
        config["nb_repeat"].append(r)
        config["nb_conv"].append(conv)
        config["nb_pool"].append(pool)
        config["nb_filter"].append(nb_filter)
        config["dropout"].append(dropout_rate)
        config["activation"].append(activation_func)
    for i in range(dense_limit):
        if random.randint(0, 10) < 4:  # prevent from getting too big
            break
        hidden_layer_size = random.randint(hard_limit ** 2, hidden_layer_limit)
        dropout_rate = random.random() * dropout_limit
        layer_limit = random.randint(2, 4)
        r = 1 + random.randint(0, layer_limit - 1)
        activation_func = dense_activation_functions[random.randint(0, len(dense_activation_functions) - 1)]
        config["nb_repeat"].append(r)
        config["dense_layer_size"].append(hidden_layer_size)
        config["dropout"].append(dropout_rate)
        config["activation"].append(activation_func)
    config["nb_classes"] = nb_classes
    config["activation"].append(final_activation)
    loss_function = loss_functions[random.randint(0, len(loss_functions) - 1)]
    optimizer = optimizers[random.randint(0, len(optimizers) - 1)]
    config["loss_function"] = loss_function
    config["optimizer"] = optimizer
    return config

cnn_dict=random_cnn_config()
print(cnn_dict)


def construct_cnn(dict_config):
    """"compile cnn from config dictionary
    :param dict_config: contains the cnn hyperparameters
    example config: {'optimizer': 'adadelta', 'nb_pool': [3, 2], 'activation': ['relu', 'relu', 'relu', 'softmax'],
    'img_cols': 28, 'dropout': [0.07861946620850124, 0.01044751378147043], 'border_mode': 'same',
    'dense_layer_size': [], 'nb_classes': 47, 'nb_filter': [26, 38], 'loss_function': 'categorical_crossentropy',
    'nb_repeat': [3, 2], 'nb_conv': [5, 5, 6], 'img_rows': 28}
    :return: keras model of type Sequential
    """
    model = Sequential()
    nb_filters = dict_config["nb_filter"]
    border_mode = dict_config["border_mode"]
    nb_convs = dict_config["nb_conv"]
    img_rows = dict_config["img_rows"]
    img_cols = dict_config["img_cols"]
    activation_funcs = dict_config["activation"]
    nb_pools = dict_config["nb_pool"]
    nb_repeats = dict_config["nb_repeat"]
    dropout_rates = dict_config["dropout"]
    dense_layer_sizes = dict_config["dense_layer_size"]
    i = 0
    model.add(Convolution2D(nb_filters[i], nb_convs[i], nb_convs[i],
                            border_mode=border_mode,
                            input_shape=(1, img_rows, img_cols)))
    model.add(Activation(activation_funcs[i]))
    for j in range(len(nb_pools)):
        i += 1
        add_convolution(model, nb_filters[i], nb_convs[i], nb_pools[i - 1], dropout_rates[i - 1], activation_funcs[i],
                        nb_repeats[i - 1])
    model.add(Flatten())
    for j, k in enumerate(dense_layer_sizes):
        i += 1
        add_dense(model, k, activation_funcs[i], dropout_rates[i - 1], nb_repeats[i - 1])
    i += 1
    model.add(Dense(dict_config["nb_classes"]))
    model.add(Activation(activation_funcs[i]))
    model.compile(loss=dict_config["loss_function"], optimizer=dict_config["optimizer"])
    return model
model= construct_cnn(cnn_dict)