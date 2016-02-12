"""
Created on Nov 18, 2015

@author: agp
"""
import os
import unittest
# from minstrel.nutils import float_or_nan, int_or_nan
# import pickle
from keras.models import Sequential

"""careful: unittests does not necessarily have to wait for previous ones to be terminated"""


class TestTrainer(unittest.TestCase):
    def __initFileParameters(self, labels=[], train_data=[], test_data=[], output_count=1):
        self.labels = labels
        self.train_data = train_data
        self.external_test_input = test_data
        self.output_count = output_count

    def __createFiles(self):
        with open("__temp_train.csv", "w") as train_csv, open("__temp_test.csv", "w") as test_csv:
            train_csv.write(",".join(self.labels) + "\n")
            train_csv.write("\n".join(self.train_data))
            test_csv.write(",".join(self.labels[self.output_count:]) + "\n")
            test_csv.write("\n".join(self.external_test_input))

    def __deleteFiles(self):
        os.remove("__temp_train.csv")
        os.remove("__temp_test.csv")
        pass

    def setUp(self):  # create a models and a temporary csv file
        self.dl_model = Sequential()

        class _: pass

        self.tp = _()
        self.tp.nb_classes = 62
        self.tp.nb_epoch = 5
        self.tp.input_dimX = 28
        self.tp.input_dimY = 28

        self.__initFileParameters(labels=["label1", "label2", "feature1", "", "feature3"],
                                  train_data=["1,bloated,,,199.7", "25.5,,0.0001,hi,123", "7,,12323.335,,1",
                                              "51,meow,0.5,lo,323"],
                                  test_data=["5,,1", "0.8,momo,3", "0.1,12,33"], output_count=2
                                  )
        self.__createFiles()

    def tearDown(self):  # delete the temporary file
        self.__deleteFiles()



if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.testName']
    # logging.basicConfig(level=logging.INFO)
    # logging.info('Started')
    unittest.main()
    # logging.info('Finished')
