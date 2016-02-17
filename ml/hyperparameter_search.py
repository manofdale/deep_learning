from __future__ import absolute_import
from __future__ import print_function

import numpy as np

from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.models import Sequential
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
from keras.utils import np_utils
import pickle

from ml import cnn_model
from ml.trainer import Trainer
from util import misc

import random


class MyModelCheckpoint(ModelCheckpoint):
    def __init__(self, filepath, best_of_the_bests=np.inf, monitor='val_acc', verbose=0,
                 save_best_only=False, mode='auto', lr_divide=3.0, patience=3.0):
        super(MyModelCheckpoint, self).__init__(filepath, monitor=monitor, verbose=verbose,
                                                save_best_only=save_best_only, mode=mode)
        self.old_best = self.best
        self.lr_divide = lr_divide
        self.best_of_the_bests = best_of_the_bests
        self.patience_ctr = 0
        self.patience=patience

    def on_epoch_end(self, epoch, logs={}):
        super(MyModelCheckpoint, self).on_epoch_end(epoch, logs=logs)
        if self.old_best != self.best:  # new best
            self.patience_ctr = 0
            self.old_best = self.best
            if self.monitor_op(self.best, self.best_of_the_bests):
                print("best %s of the bests with %f" % (self.monitor, self.best))
                self.model.save_weights("data/models/best_of_the_bests.hdf5", overwrite=True)
        else:
            self.patience_ctr += 1
            if self.patience_ctr > self.patience:
                lr = self.model.optimizer.get_config()["lr"]
                if lr < 0.001:
                    K.set_value(self.model.optimizer.lr, random.random())
                else:
                    K.set_value(self.model.optimizer.lr, lr / self.lr_divide)
                if self.verbose > 0:
                    print("decreasing learning rate %f to %f" % (lr, lr / self.lr_divide))
                self.patience_ctr = 0


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


def random_cnn_config(img_rows=28, img_cols=28, dense_limit=3, cnn_limit=3, nb_filters=42,
                      nb_pool=1,
                      nb_conv=3, nb_classes=47, cnn_dropout_limit=0.4, dropout_limit=0.5, hidden_layer_limit=1024,
                      border_modes=['same'], optimizers=[],  # 'adadelta'],
                      loss_functions=['categorical_crossentropy'],
                      cnn_activation_functions=['relu'], dense_activation_functions=['relu'],
                      final_activation='softmax',
                      ):
    border_mode = border_modes[random.randint(0, len(border_modes) - 1)]
    conv = 2 * random.randint(0, nb_conv) + 1
    nb_filter = 10 + random.randint(0, nb_filters)
    activation_func = cnn_activation_functions[random.randint(0, len(cnn_activation_functions) - 1)]
    config = {"nb_conv": [conv], "border_mode": border_mode, "img_rows": img_rows, "img_cols": img_cols,
              "nb_classes": nb_classes,
              "nb_pool": [], "nb_filter": [nb_filter], "dropout": [],
              "activation": [activation_func], "dense_layer_size": [], "nb_repeat": []}
    hard_limit = min(img_cols, img_rows)
    cnn_layer_n = 0
    for i in range(cnn_limit):
        cnn_layer_n += 1
        conv = 2 * random.randint(0, nb_conv) + 1
        pool = 2 + random.randint(0, nb_pool)
        hard_limit //= pool
        print(hard_limit, pool, conv)
        if hard_limit < 8:
            break
        activation_func = cnn_activation_functions[random.randint(0, len(cnn_activation_functions) - 1)]
        dropout_rate = random.random() * cnn_dropout_limit * random.random()
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
        if random.randint(0, 10) < 3:  # prevent from getting too big
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
    lr = random.random() * 0.8 + 0.001
    momentum = random.random() * 0.8 + 0.12

    nesterov = random.random() < 0.5
    decay = random.random() * 1e-5
    config["sgd_lr_init"] = lr
    config["sgd_momentum"] = momentum
    config["sgd_nesterov"] = nesterov
    config["sgd_decay"] = decay
    # TODO add other configs
    ''' sgd = SGD(lr=dict_config["sgd_lr_init"],
              momentum=dict_config["sgd_momentum"],
              decay=dict_config["sgd_decay"],
              nesterov=dict_config["sgd_nesterov"])'''
    # optimizers.append(sgd)
    # optimizer = optimizers[random.randint(0, len(optimizers) - 1)]
    # config["optimizer"] = optimizer
    config["loss_function"] = loss_function
    return config


def construct_cnn(dict_config):
    """"compile cnn from config dictionary
    :param dict_config: contains the cnn hyperparameters
    example config: {'optimizer': 'adadelta', 'nb_pool': [3, 2], 'activation': ['relu', 'relu', 'relu', 'softmax'],
    'img_cols': 28, 'dropout': [0.07861946620850124, 0.01044751378147043], 'border_mode': 'same',
    'dense_layer_size': [], 'nb_classes': 47, 'nb_filter': [26, 38], 'loss_function': 'categorical_crossentropy',
    'nb_repeat': [3, 2], 'nb_conv': [5, 5, 6], 'img_rows': 28}
    :return: keras models of type Sequential
    """
    print(dict_config)
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
    print(1, img_rows, img_cols)
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
    # try:
    model.add(Dense(dict_config["nb_classes"]))
    model.add(Activation(activation_funcs[i]))
    # sgd = SGD(lr=lr, decay=1e-6, momentum=0.9, nesterov=True)
    sgd = SGD(lr=dict_config["sgd_lr_init"],
              momentum=dict_config["sgd_momentum"],
              decay=dict_config["sgd_decay"],
              nesterov=dict_config["sgd_nesterov"])
    model.compile(loss=dict_config["loss_function"], optimizer=sgd)
    return model
    # except:
    #    print("the model configuration wasn't good:")
    #    print(dict_config)
    #    return None


def random_search():
    class Pack:
        pass

    tp = Pack()
    tp.nb_epoch = 50
    tp.input_dimX = 28
    tp.input_dimY = 28
    tp.nb_classes = 46 + 1  # +1 for "don't know = * " class
    train_csv = "data/dataset/all_combined_diluted.csv"
    prep = ImageDataGenerator(
        featurewise_center=True,  # set input mean to 0 over the data
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=True,  # divide inputs by std of the data
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=30,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=False,  # don't horizontally flip images
        vertical_flip=False)  # don't vertically flip images
    my_trainer = Trainer(train_csv=train_csv, test_csv=None,
                         converters=None, nan_handlers=None, empty_str_handlers=None, training_parameters=tp,
                         preprocessor=prep)
    best_of_the_bests = -np.inf
    for i in range(0, 50):
        model = None
        for tr in range(0, 50):
            dict_config = random_cnn_config()
            model = construct_cnn(dict_config)
            if model is not None:
                break
        if model is None:
            print("couldn't find a valid random configuration for the given parameters")
            break
        pickle.dump(dict_config, open("data/models/random_cnn_config_%d.p" % i, "wb"))
        print(dict_config)
        '''save_best = MyModelCheckpoint(filepath="data/models/random_cnn_config_%d_best.hdf5" % i,
                                      best_of_the_bests=best_of_the_bests, verbose=1,
                                      save_best_only=True)
                                      '''
        save_best = MyModelCheckpoint(filepath="data/models/random_cnn_config_%d_best.hdf5" % i,
                                      best_of_the_bests=best_of_the_bests, verbose=1,
                                      save_best_only=True,patience=5)
        early_stop = EarlyStopping(monitor='val_acc', patience=10, verbose=0, mode='auto')
        my_trainer.prepare_for_training(model=model, reshape_input=cnn_model.reshape_input,
                                        reshape_output=cnn_model.reshape_str_output)

        score = my_trainer.train(callbacks=[save_best, early_stop])
        print("end of training %d" % i)
        print(score)
