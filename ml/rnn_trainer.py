import logging

from keras.models import Sequential

from ml import model

#TODO update this
def batch_train(trainer, model_name_to_load, model_name_to_save, nb_epoch=10, **kwargs):
    # if not os.path.isfile("dodo.csv"):
    # self.__merge_datasets()
    # self.__check_file()
    dl_model = Sequential()
    print("segmenter batch train starts..")
    model.prepare_model3(model=dl_model, nb_classes=trainer.training_parameters.nb_classes, hidden_layers=[512],
                         **kwargs)
    if model_name_to_save is None:
        logging.warning("model is running for the first time")
    elif isinstance(model_name_to_load, basestring):
        dl_model.load_weights(model_name_to_load)

    trainer.prepare_for_training(model=dl_model, reshape_input=model.reshape_input,
                                 reshape_output=model.reshape_str_output)
    score = trainer.train()
    print("end of batch training")
    print(score)  # precision?, accuracy
    trainer.model.save_weights(model_name_to_save, overwrite=True)
    return score