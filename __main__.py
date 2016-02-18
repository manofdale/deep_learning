from ml import hyperparameter_search


class Pack:
    def __init__(self):
        self.scores = []
        self.configs = []


trainer = hyperparameter_search.init_trainer()
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
        print(meta.configs[-1])
        meta.scores.append((0,0))

    print("**************************** updating the dataset file ******************************************")
    with open("data/dataset/meta", "a") as meta_file:
        for s, c in zip(meta.scores, meta.configs):
            meta_file.write(str(s) + ":")
            meta_file.write(str(c) + "\n")
