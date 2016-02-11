import glob
import logging
import random

from keras.callbacks import ModelCheckpoint, Callback
from keras import backend as K
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from ml import cnn_model
from ml.trainer import Trainer
import numpy as np

PATH = '/home/agp/workspace/deep_learning/models/'
FILE_NAME_PREFIX = 'combined_and_defaulted_512_7_cnn_model_'
DATASET_PATH="/home/agp/workspace/deep_learning/datasets/all_combined.csv"
DROPOUT_RATES = [0.015, 0.015, 0.4]


class LReduce(Callback):
    '''change the learning rate according to the loss history

    For example: if `filepath` is `weights.{epoch:02d}-{val_loss:.2f}.hdf5`,
    then multiple files will be save with the epoch number and
    the validation loss.

    # Arguments
        monitor: quantity to monitor.
        verbose: verbosity mode, 0 or 1.
    '''

    def __init__(self, verbose=0, patience=3, lr_divide=1.2):
        super(Callback, self).__init__()
        self.verbose = verbose
        self.loss_history = []
        self.best_loss = np.Inf
        self.previous = np.Inf
        self.patience = patience
        self.wait = 0.0
        self.lr_divide = lr_divide

    def on_epoch_end(self, epoch, logs={}):
        global DROPOUT_RATES
        assert hasattr(self.model.optimizer, 'lr'), \
            'Optimizer must have a "lr" attribute.'
        current = logs.get('val_loss')
        if not np.less(current, self.previous):
            if self.wait > self.patience:
                self.wait = 0.0
                lr = self.model.optimizer.get_config()["lr"]
                print(lr, type(lr))
                if self.verbose > 0:
                    print("decreasing learning rate %f to %f" % (lr, lr / 1.01))
                K.set_value(self.model.optimizer.lr, lr / self.lr_divide)
                K.set_value(self.model.drop)
            else:
                self.wait += 1
                print("increasing dropout rates: " + ",".join([str(i) for i in DROPOUT_RATES]))
                for i, j in enumerate(DROPOUT_RATES):
                    DROPOUT_RATES[i] = j * 1.05
                print("new dropout rates: " + ",".join([str(i) for i in DROPOUT_RATES]))
        else:
            self.wait = 0.0
            if np.less(current, self.best_loss):
                lr = self.model.optimizer.get_config()["lr"]
                print(lr, type(lr))
                K.set_value(self.model.optimizer.lr, lr * 1.01)
                print("increasing learning rate from %f to %f" % (lr, lr / 1.05))
                print("decreasing dropout rates: " + ",".join([str(i) for i in DROPOUT_RATES]))
                for i, j in enumerate(DROPOUT_RATES):
                    DROPOUT_RATES[i] = j / 1.05
                print("new dropout rates: " + ",".join([str(i) for i in DROPOUT_RATES]))
            elif self.verbose > 0:
                print("learning rate is good for now")

        self.previous = current


class LossHistory(Callback):
    def __init__(self):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))


def train_forever():
    global DROPOUT_RATES
    train_csv = DATASET_PATH
    nb_epoch = 100
    current_i = 0
    start_over = False

    class Pack:
        pass

    tp = Pack()
    tp.nb_epoch = nb_epoch
    tp.input_dimX = 28
    tp.input_dimY = 28
    tp.nb_classes = 46 + 1  # +1 for "don't know = * " class
    cnn_drop = 0.01
    dense_drop = 0.1
    max_cnn_drop = 0.4
    max_dense_drop = 0.5
    prep = ImageDataGenerator(
            featurewise_center=True,  # set input mean to 0 over the dataset
            samplewise_center=False,  # set each sample mean to 0
            featurewise_std_normalization=True,  # divide inputs by std of the dataset
            samplewise_std_normalization=False,  # divide each input by its std
            zca_whitening=False,  # apply ZCA whitening
            rotation_range=20,  # randomly rotate images in the range (degrees, 0 to 180)
            width_shift_range=0.2,  # randomly shift images horizontally (fraction of total width)
            height_shift_range=0.2,  # randomly shift images vertically (fraction of total height)
            horizontal_flip=False,  # randomly flip images
            vertical_flip=False)  # randomly flip images
    my_trainer = Trainer(train_csv=train_csv, test_csv=None,
                         converters=None, nan_handlers=None, empty_str_handlers=None, training_parameters=tp,
                         preprocessor=prep)
    files = glob.glob(PATH)
    if files is None or len(files) == 0 or start_over:
        batch_train(my_trainer, None, PATH + FILE_NAME_PREFIX + '%d_epoch.hdf5' % nb_epoch, dropout_rates=DROPOUT_RATES)
        current_i = nb_epoch
    old_score = [10000, 0]
    while True:
        for i in range(current_i, 1999, nb_epoch):
            print("epoch:%d" % i)
            if i == 0 or random.random(5) < 2:
                model_name_to_load = PATH + FILE_NAME_PREFIX + "_best.hdf5"  # + str(i) + '_epoch.hdf5'
            else:
                model_name_to_load = PATH + FILE_NAME_PREFIX + str(i) + '_epoch.hdf5'
            model_name_to_save = PATH + FILE_NAME_PREFIX + str(i + nb_epoch) + '_epoch.hdf5'
            print("old DROPOUT_RATES: " + ",".join([str(x) for x in DROPOUT_RATES]))
            score = batch_train(my_trainer, model_name_to_load, model_name_to_save, nb_epoch=nb_epoch,
                                dropout_rates=DROPOUT_RATES)
            if score[1] - old_score[1] < 0.0:  # not enough momentum
                current_i = i
                if score[1] - old_score[1] < -0.05:
                    current_i = i - nb_epoch
                if current_i < 15:
                    current_i = 15
                if random.randint(0, 10) < 5:
                    nb_epoch += 1
                    tp.nb_epoch = nb_epoch
                    my_trainer.training_parameters.nb_epoch = nb_epoch
                    print("number of epochs=%d" % nb_epoch)
                max_cnn_drop += 0.005
                cnn_drop += 0.001

                if max_cnn_drop > 0.3:
                    max_cnn_drop = 0.1
                if cnn_drop > max_cnn_drop:
                    cnn_drop = max_cnn_drop / 2.0
                max_dense_drop += 0.01
                dense_drop += 0.0015
                if max_dense_drop > 0.75:
                    max_dense_drop = 0.25
                if dense_drop > max_dense_drop:
                    dense_drop = max_dense_drop / 2.0
                # random regularization param search
                DROPOUT_RATES = [random.uniform(cnn_drop, max_cnn_drop / 3), random.uniform(cnn_drop, max_cnn_drop / 4),
                                 random.uniform(dense_drop, max_dense_drop)]
                print("new DROPOUT_RATES: " + ",".join([str(x) for x in DROPOUT_RATES]))
                print(nb_epoch)
                break
            old_score = score


def batch_train(my_trainer, model_name_to_load, model_name_to_save, nb_epoch=10, **kwargs):
    dl_model = Sequential()
    print("batch train starts..")
    cnn_model.prepare_model7(model=dl_model, nb_classes=my_trainer.training_parameters.nb_classes, hidden_layers=[2048],
                             **kwargs)
    if model_name_to_save is None:
        logging.warning("model is running for the first time")
    elif isinstance(model_name_to_load, basestring):
        dl_model.load_weights(model_name_to_load)

    my_trainer.prepare_for_training(model=dl_model, reshape_input=cnn_model.reshape_input,
                                    reshape_output=cnn_model.reshape_str_output)
    save_best = ModelCheckpoint(filepath=PATH + FILE_NAME_PREFIX + "_best.hdf5", verbose=1, save_best_only=True)
    adjust_learning_rate = LReduce(verbose=1, patience=3, lr_divide=1.2)
    score = my_trainer.train(callbacks=[save_best])
    print("end of batch training")
    print(score)
    my_trainer.model.save_weights(model_name_to_save, overwrite=True)
    return score


train_forever()
