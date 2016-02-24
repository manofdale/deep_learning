from ml import hyperparameter_search


class Pack:
    def __init__(self):
        self.scores = []
        self.configs = []


trainer = hyperparameter_search.init_trainer()
trainer.training_parameters.nb_epoch=500
dict_config = {'bias_regularizers': [None, None, None, None], 'nb_pool': [3], 'nb_conv': [5, 3],
               'sgd_momentum': 0.04509058440772208, 'img_cols': 28, 'sgd_decay': 3.970867037476346e-06,
               'dense_weight_regularizers': [None, None, None, ('l2', 0.00590578693693914)], 'border_mode': 'valid',
               'sgd_lr_divide': 1.6892790823754182, 'dense_layer_size': [749, 879, 254, 864],
               'loss_function': 'categorical_crossentropy', 'nb_classes': 47, 'sgd_lr_init': 0.17006898785437682,
               'dense_inits': ['uniform', 'glorot_uniform', 'uniform', 'glorot_uniform'], 'sgd_nesterov': False,
               'nb_filter': [32, 47], 'activation': ['linear', 'relu', 'linear', 'relu', 'relu', 'relu', 'softmax'],
               'nb_repeat': [2, 2, 2, 2, 1],
               'dropout': [0.05581567092442845, 0.246871561047429298, 0.0575979640541265788, 0.4836863105510306,
                           0.04954334377523167], 'dense_activity_regularizers': [None, None, None, None],
               'img_rows': 28}
meta = Pack()
hyperparameter_search.search_near_promising(meta, trainer, dict_config, "48151")
with open("data/dataset/meta_near_48151", "a") as meta_file:
        for s, c in zip(meta.scores, meta.configs):
            meta_file.write(str(s) + ":")
            meta_file.write(str(c) + "\n")
trainer.training_parameters.nb_epoch=50
raw_input("tests ended, presse enter to continue")
for i in range(2000):
    print(i)
    meta = Pack()
    try:
        hyperparameter_search.random_search(meta, trainer)
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception as e:
        print("this config caused an exception:")
        print(e)
        import traceback

        print(traceback.format_exc())
        print(meta.configs[-1])
        meta.scores.append((0, 0))

    print("**************************** updating the dataset file ******************************************")
    with open("data/dataset/meta", "a") as meta_file:
        for s, c in zip(meta.scores, meta.configs):
            meta_file.write(str(s) + ":")
            meta_file.write(str(c) + "\n")
