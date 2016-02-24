from __future__ import absolute_import
from __future__ import print_function

import numpy as np
import pickle
import random

from keras import backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.models import Sequential
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2, activity_l2, l1, activity_l1, l1l2, activity_l1l2

from ml import cnn_model
from ml.trainer import Trainer
from util import misc
from util.misc import f_or_default


def append_to_dataset(dataset, score, dict_config):
    """for now"""
    dataset.append([score + 1, dict_config])


def check_roughly_monotonic(ar, limit_n=3, best=-np.inf, cmp=lambda x, y: x < y):
    """check whether given list of numbers is roughly increasing

    :param n: is there a better value than the max within n steps
    :return True if there is
    """
    i = 0
    steps = 0.0
    k = 0.0
    while i < len(ar):
        doing_well = False
        k += 1.0
        for j in range(limit_n):
            if i >= len(ar):
                doing_well = True  # survived this far so it's ok
                steps += j
                break
            if cmp(best, ar[i]):
                best = ar[i]
                doing_well = True
                steps += j
                continue
            i += 1
        if doing_well is False:
            break
    steps /= k

    print("this config has the performance: %s" % best)
    return doing_well, steps, best


def score_training_history(log_history, patience=5):  # patience is the number of classes
    """return an integer score, the bigger the better"""
    # grad=misc.gradient_1d(log_history)
    good, score, best = check_roughly_monotonic(log_history, limit_n=patience)
    # TODO add other ways to check the history and combine the scores
    if good:
        return 1000 - int(score), best
    else:
        return int(score) + 1, best  # make sure that it is never zero


class MyModelCheckpoint(ModelCheckpoint):
    def __init__(self, filepath, best_of_the_bests=np.inf, monitor='val_acc', verbose=0,
                 save_best_only=False, mode='auto', lr_divide=3.0, drop_divide=0.975, patience=3.0):
        super(MyModelCheckpoint, self).__init__(filepath, monitor=monitor, verbose=verbose,
                                                save_best_only=save_best_only, mode=mode)
        self.old_best = self.best
        self.lr_divide = lr_divide
        self.best_of_the_bests = best_of_the_bests
        self.patience_ctr = 0
        self.patience = patience
        self.drop_divide = drop_divide
        self.log_history = []

    def on_epoch_end(self, epoch, logs={}):
        super(MyModelCheckpoint, self).on_epoch_end(epoch, logs=logs)
        self.log_history.append(logs.get(self.monitor))
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
                    lr = random.random() * 0.2
                    print("assigning new random learning rate:%f" % lr)
                    K.set_value(self.model.optimizer.lr, lr)
                else:
                    K.set_value(self.model.optimizer.lr, lr / self.lr_divide)
                if self.verbose > 0:
                    print("decreasing learning rate %f to %f" % (lr, lr / self.lr_divide))
                self.patience_ctr = 0


def add_convolution(model, nb_filters, nb_conv, nb_pool, dropout_r, activation_func, r, old_model=None, k=0, k_lim=0):
    for i in range(r):  # no more than limit
        if old_model is not None and k + i < k_lim:
            model.add(Convolution2D(nb_filters, nb_conv, nb_conv, weights=old_model.layers[k + i].get_weights()))
        else:
            model.add(Convolution2D(nb_filters, nb_conv, nb_conv))
        model.add(Activation(activation_func))
    model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
    model.add(Dropout(dropout_r))


def add_dense(model, hidden_layer_size, activation_function, dropout_rate, r, act_r=(l2, 0.01),
              weight_r=(l2, 0.01), b_r=(l2, 0.01), init='glorot_uniform', old_model=None, k=0, k_lim=0):
    """

    :param model:
    :param hidden_layer_size:
    :param activation_function:
    :param dropout_rate:
    :param r:
    :param act_r:
    :param weight_r:
    :param b_r:
    :param init:
    :param old_model: optionally init weight from here
    :param k:
    :param k_lim: after current layer number k+i >= k_lim, do not init with old weights
    """
    for i in range(r):
        if old_model is not None and k + i < k_lim:
            model.add(Dense(hidden_layer_size, weights=old_model.layers[k + i].get_weights(),
                            W_regularizer=f_or_default(weight_r),
                            activity_regularizer=f_or_default(act_r), b_regularizer=f_or_default(b_r)))
        else:
            model.add(Dense(hidden_layer_size, init=init, W_regularizer=f_or_default(weight_r),
                            activity_regularizer=f_or_default(act_r), b_regularizer=f_or_default(b_r)))

        # model.layers[-1].regularizers[-1].set_param(model.layers[-1].get_params()[0][0])
        model.add(Activation(activation_function))
        model.add(Dropout(dropout_rate))


def random_cnn_config(img_rows=28, img_cols=28, dense_limit=4, cnn_limit=6, nb_filters=42,
                      nb_pool=2,
                      nb_conv=3, nb_classes=47, lr_limit=1.5, momentum_limit=0.9, cnn_dropout_limit=0.5,
                      dropout_limit=0.5, hidden_layer_limit=1024,
                      inits=['zero', 'glorot_uniform', 'normal', 'glorot_uniform', 'uniform', 'glorot_uniform',
                             'he_uniform', 'he_normal', 'glorot_normal', 'glorot_uniform', 'he_normal',
                             'glorot_uniform', 'he_uniform', 'glorot_uniform', 'glorot_uniform', 'glorot_uniform', ],
                      border_modes=['same', 'valid', 'same', 'same'], optimizers=[],  # 'adadelta'],
                      loss_functions=['categorical_crossentropy'],
                      cnn_activation_functions=['relu', 'tanh', 'relu', 'relu', 'hard_sigmoid', 'relu', 'linear',
                                                'relu', ],
                      dense_activation_functions=['relu', 'relu', 'relu', 'hard_sigmoid', 'relu', 'linear', 'relu',
                                                  'relu', ],
                      regularizers=['l1', 'l2', 'l2', 'l2', 'l1l2', 'l2', 'l2', 'l2'],
                      activity_regularizers=['activity_l1', 'activity_l2', 'activity_l1l2', 'activity_l2',
                                             'activity_l2', ],
                      final_activation='softmax', finalize=True, random_dense=True, random_cnn=True, config=None,
                      init_config=True,
                      ):
    cnn_layer_n = 0
    hard_limit = min(img_rows, img_cols)
    if init_config:
        border_mode = border_modes[random.randint(0, len(border_modes) - 1)]
        conv = 2 * random.randint(0, nb_conv) + 1
        nb_filter = 10 + random.randint(0, nb_filters)
        activation_func = cnn_activation_functions[random.randint(0, len(cnn_activation_functions) - 1)]
        config = {"nb_conv": [conv], "border_mode": border_mode, "img_rows": img_rows, "img_cols": img_cols,
                  "nb_classes": nb_classes, "dense_inits": [], "dense_weight_regularizers": [],
                  "dense_activity_regularizers": [],
                  "bias_regularizers": [],
                  "nb_pool": [], "nb_filter": [nb_filter], "dropout": [],
                  "activation": [activation_func], "dense_layer_size": [], "nb_repeat": []}
        hard_limit = min(img_cols, img_rows)

    else:  # infer params from the config
        for j in range(len(config["nb_pool"])):
            cnn_layer_n += config["nb_repeat"]
    if random_cnn:
        hard_limit = random_add_cnn_to_config(config, cnn_limit, cnn_layer_n, nb_conv, nb_filters, nb_pool, hard_limit,
                                              cnn_activation_functions, cnn_dropout_limit)
    else:
        for i in config["nb_pool"]:
            hard_limit //= i
    if random_dense:
        random_add_dense_to_config(config, dense_limit, hard_limit, hidden_layer_limit, inits,
                                   dense_activation_functions,
                                   regularizers, dropout_limit, activity_regularizers)
    if finalize:
        finalize_config(config, nb_classes, final_activation, loss_functions, lr_limit, momentum_limit)
    return config


def random_add_cnn_to_config(config, cnn_limit, cnn_layer_n, nb_conv, nb_filters, nb_pool, hard_limit,
                             cnn_activation_functions, cnn_dropout_limit):
    for i in range(cnn_limit):
        cnn_layer_n += 1
        conv = 2 * random.randint(0, nb_conv) + 1
        pool = 2 + random.randint(0, nb_pool)
        hard_limit //= pool
        print(hard_limit, pool, conv)
        if random.random() < 0.2 or hard_limit < 4:
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
    return hard_limit


def random_add_dense_to_config(config, dense_limit, hard_limit, hidden_layer_limit, inits, dense_activation_functions,
                               regularizers, dropout_limit, activity_regularizers):
    for i in range(dense_limit):
        if random.randint(0, 20) < 3:  # prevent from getting too big
            break
        hidden_layer_size = random.randint(hard_limit ** 2, hidden_layer_limit)
        dropout_rate = random.random() * dropout_limit
        layer_limit = random.randint(2, 4)
        r = 1 + random.randint(0, layer_limit - 1)
        init = inits[random.randint(0, len(inits) - 1)]
        config["dense_inits"].append(init)
        activation_func = dense_activation_functions[random.randint(0, len(dense_activation_functions) - 1)]
        config["nb_repeat"].append(r)
        config["dense_layer_size"].append(hidden_layer_size)
        if random.random() < 0.1:
            reg = regularizers[random.randint(0, len(regularizers) - 1)]
            if reg == l1l2:
                config["dense_weight_regularizers"].append((reg, random.random() * 0.1, random.random() * 0.1))
            else:
                config["dense_weight_regularizers"].append((reg, random.random() * 0.1))
        else:
            config["dense_weight_regularizers"].append(None)
        if random.random() < 0.1:
            reg = regularizers[random.randint(0, len(regularizers) - 1)]
            if reg == l1l2:
                config["bias_regularizers"].append((reg, random.random() * 0.1, random.random() * 0.1))
            else:
                config["bias_regularizers"].append((reg, random.random() * 0.1))
        else:
            config["bias_regularizers"].append(None)
        if random.random() < 0.1:
            reg = activity_regularizers[random.randint(0, len(activity_regularizers) - 1)]
            if reg == activity_l1l2:
                config["dense_activity_regularizers"].append((reg, random.random() * 0.1, random.random() * 0.1,))
            else:
                config["dense_activity_regularizers"].append((reg, random.random() * 0.1))
        else:
            config["dense_activity_regularizers"].append(None)
        config["dropout"].append(dropout_rate)
        config["activation"].append(activation_func)


def finalize_config(config, nb_classes, final_activation, loss_functions, lr_limit, momentum_limit):
    config["nb_classes"] = nb_classes
    config["final_activation"] = final_activation
    loss_function = loss_functions[random.randint(0, len(loss_functions) - 1)]
    lr = random.random() * lr_limit * random.random() + 0.001
    momentum = random.random() * momentum_limit * random.random() + 0.01
    nesterov = random.random() < 0.5
    decay = random.random() * 1e-5
    config["sgd_lr_init"] = lr
    config["sgd_momentum"] = momentum
    config["sgd_nesterov"] = nesterov
    config["sgd_decay"] = decay
    config["sgd_lr_divide"] = 1 + random.random() * 3
    config["loss_function"] = loss_function
    # TODO add other configs
    ''' sgd = SGD(lr=dict_config["sgd_lr_init"],
              momentum=dict_config["sgd_momentum"],
              decay=dict_config["sgd_decay"],
              nesterov=dict_config["sgd_nesterov"])'''
    # optimizers.append(sgd)
    # optimizer = optimizers[random.randint(0, len(optimizers) - 1)]
    # config["optimizer"] = optimizer


def str_to_regularizer(r):
    """ :return the regularizer function with its arguments according to the string r

     sample r: ('l1',0.05)
     another sample r: ('l1l2',0.01,0.02)
     """
    if r is None:
        return None
    if r[0] == 'l1':
        return misc.expand_args(l1, *r[1:])
    if r[0] == 'l2':
        return misc.expand_args(l2, *r[1:])
    if r[0] == 'l1l2':
        return misc.expand_args(l1l2, *r[1:])
    if r[0] == 'activity_l1':
        return misc.expand_args(activity_l1, *r[1:])
    if r[0] == 'activity_l2':
        return misc.expand_args(activity_l2, *r[1:])
    if r[0] == 'activity_l1l2':
        return misc.expand_args(activity_l1l2, *r[1:])
    return None


def construct_cnn(dict_config, old_model=None, k_lim=0):
    """"compile cnn from config dictionary, optionally get initials weights of the earlier layers from an older model

    :param dict_config: contains the cnn hyperparameters
    :param old_model: an older model that can be used when constructing the new one
    :param k_lim: number of layers in the old model from which the weights will be directly taken

    sample dict_config: (995, 0.4815135402497836):{'bias_regularizers': [None, None, None, None], 'nb_pool': [3],
    'nb_conv': [5, 3], 'sgd_momentum': 0.04509058440772208, 'img_cols': 28, 'sgd_decay': 3.970867037476346e-06,
    'dense_weight_regularizers': [None, None, None, ('l2', 0.00590578693693914)], 'border_mode': 'valid',
    'sgd_lr_divide': 1.6892790823754182, 'dense_layer_size': [749, 879, 254, 864],
    'loss_function': 'categorical_crossentropy', 'nb_classes': 47, 'sgd_lr_init': 0.17006898785437682,
    'dense_inits': ['uniform', 'glorot_uniform', 'uniform', 'glorot_uniform'], 'sgd_nesterov': False,
    'nb_filter': [32, 47], 'activation': ['linear', 'relu', 'linear', 'relu', 'relu', 'relu', 'softmax'],
    'nb_repeat': [2, 2, 2, 2, 1], 'dropout': [0.0558, 0.0687, 0.1598, 0.0208, 0.4795],
    'dense_activity_regularizers': [None, None, None, None], 'img_rows': 28}

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
    inits = dict_config["dense_inits"]
    i = 0
    layer_k = 0
    model.add(Convolution2D(nb_filters[i], nb_convs[i], nb_convs[i],
                            border_mode=border_mode,
                            input_shape=(1, img_rows, img_cols)))
    layer_k += 1
    print(1, img_rows, img_cols)
    model.add(Activation(activation_funcs[i]))
    for j in range(len(nb_pools)):
        i += 1
        add_convolution(model, nb_filters[i], nb_convs[i], nb_pools[i - 1], dropout_rates[i - 1], activation_funcs[i],
                        nb_repeats[i - 1], old_model=old_model, k=layer_k, k_lim=k_lim)
        layer_k += nb_repeats[i - 1]
    model.add(Flatten())
    act_rs = misc.convert_xs(dict_config["dense_activity_regularizers"], str_to_regularizer)
    weight_rs = misc.convert_xs(dict_config["dense_weight_regularizers"], str_to_regularizer)
    b_rs = misc.convert_xs(dict_config["bias_regularizers"], str_to_regularizer)

    for j, k in enumerate(dense_layer_sizes):
        i += 1
        add_dense(model, k, activation_funcs[i], dropout_rates[i - 1], nb_repeats[i - 1], act_rs[j], weight_rs[j],
                  b_rs[j], init=inits[j], old_model=old_model, k=layer_k, k_lim=k_lim)
        layer_k += nb_repeats[i - 1]
    i += 1
    # try:
    model.add(Dense(dict_config["nb_classes"]))
    model.add(Activation(dict_config["final_activation"]))
    # sgd = SGD(lr=lr, decay=1e-6, momentum=0.9, nesterov=True)
    sgd = SGD(lr=dict_config["sgd_lr_init"],
              momentum=dict_config["sgd_momentum"],
              decay=dict_config["sgd_decay"],
              nesterov=dict_config["sgd_nesterov"])
    model.compile(loss=dict_config["loss_function"], optimizer=sgd)
    return model


def init_trainer():
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
                         converters=None, nan_handlers=None,
                         empty_str_handlers=None, training_parameters=tp,
                         preprocessor=prep)
    return my_trainer


def init_best():
    import os
    best_of_the_bests = None
    if os.path.isfile("data/variables/best"):
        with open("data/variables/best", "r") as best_file:
            for i in best_file:
                try:
                    if len(i.strip()) > 0:
                        best_of_the_bests = float(i)
                except ValueError:
                    print("warning: the file data/variables/best has nan entries")
    if best_of_the_bests is None:
        best_of_the_bests = -np.inf
    return best_of_the_bests


def search_near_promising(meta, my_trainer, config, checkpoint_name, n_itr=50):
    itr = 0
    import copy
    dense_limit = 3
    hard_limit = 7
    hidden_layer_limit = 1024
    inits = ['zero', 'glorot_uniform', 'normal', 'glorot_uniform', 'uniform', 'glorot_uniform',
             'he_uniform', 'he_normal', 'glorot_normal', 'glorot_uniform', 'he_normal',
             'glorot_uniform', 'he_uniform', 'glorot_uniform', 'glorot_uniform', 'glorot_uniform', ]
    dense_activation_functions = ['relu', 'relu', 'relu', 'hard_sigmoid', 'relu', 'linear', 'relu', 'relu', ]
    regularizers = ['l1', 'l2', 'l2', 'l2', 'l1l2', 'l2', 'l2', 'l2']
    dropout_limit = 0.5
    activity_regularizers = ['activity_l1', 'activity_l2', 'activity_l1l2', 'activity_l2',
                             'activity_l2', ]

    while itr < n_itr:
        itr += 1

        dict_config = copy.deepcopy(config)  # initial config
        random_add_dense_to_config(dict_config, dense_limit, hard_limit, hidden_layer_limit, inits,
                                   dense_activation_functions, regularizers, dropout_limit, activity_regularizers)
        meta.configs.append(dict_config)
        model = construct_cnn(dict_config)

        test_patience = 10
        if model is None:
            print("something is wrong with the config")
            return
        my_trainer.prepare_for_training(model=model, reshape_input=cnn_model.reshape_input,
                                        reshape_output=cnn_model.reshape_str_output)
        best_of_the_bests = init_best()
        save_best = MyModelCheckpoint(filepath="data/models/promising_cnn_config_test_%s_%d.hdf5" % (checkpoint_name,i),
                                      best_of_the_bests=best_of_the_bests, verbose=1,
                                      save_best_only=True, patience=6, lr_divide=dict_config["sgd_lr_divide"])
        early_stop = EarlyStopping(monitor='val_acc', patience=test_patience, verbose=0, mode='auto')
        my_trainer.prepare_for_training(model=model, reshape_input=cnn_model.reshape_input,
                                        reshape_output=cnn_model.reshape_str_output)
        score = my_trainer.train(callbacks=[save_best, early_stop])
        print("end of training %s" % checkpoint_name)
        print(score)
        meta.scores.append(score_training_history(save_best.log_history, patience=test_patience))


def random_search(meta, my_trainer):
    best_of_the_bests = init_best()

    for i in range(0, 50):
        print("*********** batch:%d **********" % i)
        training_patience = 4
        model = None
        meta.configs.append([])
        ix = len(meta.configs) - 1
        for tr in range(0, 50):
            dict_config = random_cnn_config()
            meta.configs[ix] = dict_config
            model = construct_cnn(dict_config)
            if model is not None:
                break
        if model is None:
            print("couldn't find a valid random configuration for the given parameters")
            break
        pickle.dump(dict_config, open("data/models/random_cnn_config_%d.p" % i, "wb"))
        save_best = MyModelCheckpoint(filepath="data/models/random_cnn_config_%d_best.hdf5" % i,
                                      best_of_the_bests=best_of_the_bests, verbose=1,
                                      save_best_only=True, patience=1, lr_divide=dict_config["sgd_lr_divide"])
        early_stop = EarlyStopping(monitor='val_acc', patience=training_patience, verbose=0, mode='auto')
        my_trainer.prepare_for_training(model=model, reshape_input=cnn_model.reshape_input,
                                        reshape_output=cnn_model.reshape_str_output)

        score = my_trainer.train(callbacks=[save_best, early_stop])
        best_of_the_bests = save_best.best_of_the_bests
        meta.scores.append(score_training_history(save_best.log_history, patience=training_patience))
        with open("data/variables/best", "w") as best_file:
            best_file.write(str(best_of_the_bests))
        print("end of training %d" % i)
        print(score)
