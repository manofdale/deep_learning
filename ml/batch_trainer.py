import glob
import logging
import random

from keras.callbacks import ModelCheckpoint, Callback
from keras.models import Sequential

from ml import cnn_model
from ml.trainer import Trainer

PATH = '/home/agp/workspace/deep_learning/models/'
FILE_NAME_PREFIX = 'combined_and_defaulted_2048_3_cnn_model_'


class LossHistory(Callback):
    def __init__(self):
        self.losses=[]

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))


def train_forever():
    train_csv = "/home/agp/workspace/deep_learning/datasets/all_combined.csv"
    nb_epoch = 60
    current_i = 48
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
    dropout_rates = [0.25, 0.25, 0.5]
    my_trainer = Trainer(train_csv=train_csv, test_csv=None,
                         converters=None, nan_handlers=None, empty_str_handlers=None, training_parameters=tp)
    files = glob.glob(PATH)
    if files is None or len(files) == 0 or start_over:
        batch_train(my_trainer, None, PATH + FILE_NAME_PREFIX + '%d_epoch.hdf5' % nb_epoch, dropout_rates=dropout_rates)
        current_i = nb_epoch
    old_score = [10000, 0]
    while True:
        for i in range(current_i, 1999, nb_epoch):
            print("epoch:%d" % i)
            model_name_to_load = PATH + FILE_NAME_PREFIX + str(i) + '_epoch.hdf5'
            model_name_to_save = PATH + FILE_NAME_PREFIX + str(i + nb_epoch) + '_epoch.hdf5'
            print("old dropout_rates: " + ",".join([str(x) for x in dropout_rates]))
            score = batch_train(my_trainer, model_name_to_load, model_name_to_save, nb_epoch=nb_epoch,
                                dropout_rates=dropout_rates)
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
                dropout_rates = [random.uniform(cnn_drop, max_cnn_drop / 3), random.uniform(cnn_drop, max_cnn_drop / 4),
                                 random.uniform(dense_drop, max_dense_drop)]
                print("new dropout_rates: " + ",".join([str(x) for x in dropout_rates]))
                print(nb_epoch)
                break
            old_score = score


def batch_train(my_trainer, model_name_to_load, model_name_to_save, nb_epoch=10, **kwargs):
    dl_model = Sequential()
    print("batch train starts..")
    cnn_model.prepare_model3(model=dl_model, nb_classes=my_trainer.training_parameters.nb_classes, hidden_layers=[2048],
                             **kwargs)
    if model_name_to_save is None:
        logging.warning("model is running for the first time")
    elif isinstance(model_name_to_load, basestring):
        dl_model.load_weights(model_name_to_load)

    my_trainer.prepare_for_training(model=dl_model, reshape_input=cnn_model.reshape_input,
                                    reshape_output=cnn_model.reshape_str_output)
    save_best = ModelCheckpoint(filepath=PATH + FILE_NAME_PREFIX + "_best.hdf5", verbose=1, save_best_only=True)
    score = my_trainer.train(callbacks=[save_best])
    print("end of batch training")
    print(score)
    my_trainer.model.save_weights(model_name_to_save, overwrite=True)
    return score


train_forever()
