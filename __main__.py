from ml import hyperparameter_search


class Pack:
    def __init__(self):
        self.scores = []
        self.configs = []


trainer = hyperparameter_search.init_trainer()
trainer.training_parameters.nb_epoch = 5
dict_config0 = {'bias_regularizers': [None, None, None], 'nb_pool': [2, 2], 'nb_conv': [3, 1, 3],
                'final_activation': 'softmax', 'img_cols': 28, 'sgd_decay': 2.255626836893722e-06,
                'dense_weight_regularizers': [None, None, None], 'sgd_momentum': 0.43162749108853093,
                'border_mode': 'same', 'sgd_lr_divide': 3.4088286172673765, 'dense_layer_size': [344, 836, 796],
                'loss_function': 'categorical_crossentropy', 'nb_classes': 47, 'sgd_lr_init': 0.2571098030976638,
                'dense_inits': ['glorot_uniform', 'he_uniform', 'glorot_uniform'], 'sgd_nesterov': True,
                'nb_filter': [30, 35, 18], 'activation': ['relu', 'tanh', 'relu', 'relu', 'linear', 'relu'],
                'nb_repeat': [2, 2, 3, 1, 3],
                'dropout': [0.34909860728678416, 0.1068087623123284, 0.14321283579445127, 0.42233754089393727,
                            0.35701783237339146], 'dense_activity_regularizers': [None, None, None], 'img_rows': 28}

dict_config = {'bias_regularizers': [None, None], 'nb_pool': [3], 'nb_conv': [3, 3], 'sgd_momentum': 0.3204366612904804,
               'img_cols': 28, 'sgd_decay': 3.0910527065849217e-06, 'dense_weight_regularizers': [None, None],
               'border_mode': 'valid', 'sgd_lr_divide': 1.4869444378091634, 'dense_layer_size': [784, 671],
               'loss_function': 'categorical_crossentropy', 'nb_classes': 47, 'sgd_lr_init': 0.11694265856353742,
               'dense_inits': ['uniform', 'glorot_uniform'], 'sgd_nesterov': True, 'nb_filter': [16, 44],
               'activation': ['relu', 'relu', 'relu', 'relu'], 'final_activation': 'softmax', 'nb_repeat': [1, 2, 2],
               'dropout': [0.10595141800496129, 0.4739859909661407, 0.50399789654762237],
               'dense_activity_regularizers': [None, None], 'img_rows': 28}
dict_config2 = {'bias_regularizers': [None, None, None, None], 'nb_pool': [3], 'nb_conv': [5, 3],
                'sgd_momentum': 0.04509058440772208, 'img_cols': 28, 'sgd_decay': 3.970867037476346e-06,
                'dense_weight_regularizers': [None, None, None, ('l2', 0.00590578693693914)], 'border_mode': 'valid',
                'sgd_lr_divide': 1.6892790823754182, 'dense_layer_size': [749, 879, 254, 864],
                'loss_function': 'categorical_crossentropy', 'nb_classes': 47, 'sgd_lr_init': 0.17006898785437682,
                'dense_inits': ['uniform', 'glorot_uniform', 'uniform', 'glorot_uniform'], 'sgd_nesterov': False,
                'nb_filter': [32, 47], 'activation': ['linear', 'relu', 'linear', 'relu', 'relu', 'relu'],
                'nb_repeat': [2, 2, 2, 2, 1],
                'dropout': [0.05581567092442845, 0.246871561047429298, 0.0575979640541265788, 0.4836863105510306,
                            0.04954334377523167], 'dense_activity_regularizers': [None, None, None, None],
                'img_rows': 28, 'final_activation': 'softmax'}
meta = Pack()
population_configs = [(0,41,dict_config0),(0.48151, dict_config2), (0.51916, dict_config)]
hyperparameter_search.search_around_promising(meta, trainer, population_configs, 0.4, "51916")
print(population_configs)
with open("data/dataset/meta_near_48151_incremental", "a") as meta_file:
    for s, c in zip(meta.scores, meta.configs):
        meta_file.write(str(s) + ":")
        meta_file.write(str(c) + "\n")
trainer.training_parameters.nb_epoch = 50
raw_input("tests ended, press enter to continue")
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
