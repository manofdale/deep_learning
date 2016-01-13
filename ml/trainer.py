"""
Created on Nov 17, 2015

@author: agp
"""
import logging
from math import sqrt
from numpy.random import shuffle

from util import misc
from util.misc import fill_missing_data, nan_to_mean, empty_to_mode, \
    int_or_nan


# from numpy.f2py.auxfuncs import throw_error
class Trainer(object):
    """Train using a given model and dataset
    classdocs
    """

    def __init__(self, refined_rows=None, unrefined_rows=None, train_csv="train.csv", test_csv="test.csv", data_row=1,
                 output_count=1,
                 converters=[int_or_nan],
                 nan_handlers=[nan_to_mean], empty_str_handlers=[empty_to_mode],
                 model=None, training_parameters=None, supervised=True):
        """Constructor
        
        Parameters:
        @param typed_rows: contains already shuffled input, output and external test data instead of a csv file
        @param train_csv:name of the file that contains training dataset, may or may not have a line describing the labels 
        @param test_csv: name of the file that contains testing dataset, may or may not have a line describing the labels 
        @param data_row: the line number where real data starts,labels line is assumed to be given just before this line
        @param output_count: number of outputs (or classes) to be trained, give -1 to autodetect from labels
            if output_count is -1 then reassign output_count to # labels that start with 'label' 
            (e.g. n for 'label1,label2,..,labeln,input0,..,inputm') if can find,
            else assign it to zero and set the mode to unsupervised
        @param converters: an array of functions that will be used for str -> data_type conversion
            str-> number (e.g. int, float, binary, hex) conversions should handle missing data and return 'NaN'
        @param nan_handlers: convert 'NaN' into a number in a given column
        @param empty_str_handlers: convert empty string into categorical data in a given column
        @param model: model object that can be trained with model.fit(self.input_rows,self.output_rows,
        test_to_train_ratio=0.5,validation_ratio=0.2), it should randomly select given ratios of input and output rows 
        for training and validation, and the rest for testing        
        """
        self.model = model
        self.training_parameters = training_parameters
        self.supervised = supervised
        self.output_count = output_count

        if refined_rows is not None and len(refined_rows) > 4:  # already have good data, ready to use for the model
            self.labels = refined_rows[0]
            self.train_input = refined_rows[1]
            self.train_output = refined_rows[2]
            self.test_input = refined_rows[3]
            self.test_output = refined_rows[4]
            if len(refined_rows) > 5:
                self.external_reshaped_test_input = refined_rows[5]
            else:
                self.load_train_test_csv(None, test_csv, data_row, converters)
        elif unrefined_rows is not None and len(unrefined_rows) > 2:
            """input and output is already given in rows, already shuffled
            but it will require further reshaping and horizontal splitting (train/test) for the model"""
            self.labels = unrefined_rows[0]
            self.input_rows = unrefined_rows[1]
            self.output_rows = unrefined_rows[2]
            if len(unrefined_rows) > 3:
                self.external_test_input = unrefined_rows[3]
            else:
                self.load_train_test_csv(None, test_csv, data_row, converters)
        else:
            self.load_train_test_csv(train_csv, test_csv, data_row, converters)
        if nan_handlers is not None and empty_str_handlers is not None and len(refined_rows) == 0:
            if len(nan_handlers) == 1:
                nan_handlers *= len(self.labels)
            if len(empty_str_handlers) == 1:
                empty_str_handlers *= len(self.labels)
            logging.info("handling possible missing numbers")
            self.handle_missing_data(nan_handlers, empty_str_handlers)
            """@Todo check why this doesn't work when not given train_csv"""
        else:
            logging.info("assuming no missing numbers in the dataset")
        if model is not None:
            logging.info("model assigned:" + model.str())
        else:
            logging.info("no model is assigned yet")

    def check_set_parameters(self):
        """make a default one if a parameter doesn't exist"""
        batch_size = 128
        nb_epoch = 1
        show_accuracy = True
        train_to_test_ratio = 6
        verbose = 1
        if not hasattr(self.training_parameters, 'batch_size'):  # defaults
            logging.info("batch size is set to default: %d" % batch_size)
            self.training_parameters.batch_size = batch_size
        if not hasattr(self.training_parameters, 'nb_epoch'):
            logging.info("number of epochs is set to default: %d" % nb_epoch)
            self.training_parameters.nb_epoch = nb_epoch
        if not hasattr(self.training_parameters, 'show_accuracy'):
            self.training_parameters.show_accuracy = show_accuracy
        if not hasattr(self.training_parameters, 'train_to_test_ratio'):
            logging.info("split ratio is set to default: %d" % train_to_test_ratio)
            self.training_parameters.train_to_test_ratio = train_to_test_ratio
        if not hasattr(self.training_parameters, 'verbose'):
            self.training_parameters.verbose = verbose
        if not hasattr(self.training_parameters, 'input_dimX'):
            square_n = sqrt(len(self.input_rows[0]))
            if int(square_n) == square_n:
                square_n = int(square_n)
                self.training_parameters.input_dimX = square_n
                logging.warning(square_n)
                self.training_parameters.input_dimY = square_n
            else:
                logging.error("square root of input row length is not an integer, (not a square image)")
                # throw_error("Error: Missing parameter: image dimensions")
        if not hasattr(self.training_parameters, 'nb_classes'):
            self.training_parameters.nb_classes = 2  # assume binary classification

    def split_dataset(self, reshape_input=None, reshape_output=None):
        """
        @param reshape_input: function to reshape input
        @param reshape_output: function to reshape output
        @todo would I ever need to reshape output if unsupervised, 
        even if normally no output would be required?
        """
        n = len(self.input_rows)
        r = self.training_parameters.train_to_test_ratio
        train_size = (n * r) // (r + 1)
        if reshape_input is None:
            logging.warn("input data is not reshaped")
            self.train_input = self.input_rows[:train_size]
            self.test_input = self.input_rows[train_size:]
        else:
            logging.warning("reshaping")
            self.train_input = reshape_input(self.input_rows[:train_size],
                                             self.training_parameters.input_dimX,
                                             self.training_parameters.input_dimY)
            # self.train_input -= train_mean
            # self.train_input /= train_std
            self.test_input = reshape_input(self.input_rows[train_size:], self.training_parameters.input_dimX,
                                            self.training_parameters.input_dimY)
            # self.test_input -= train_mean # we don't use test split's mean and std
            # self.test_input /= train_std # we don't use test split's mean and std
        if not self.supervised:  # and len(self.output_rows)==0:# no output?
            return
        if reshape_output is None:
            logging.warn("output data is not reshaped")
            self.train_output = self.output_rows[:train_size]
            self.test_output = self.output_rows[train_size:]
        else:
            self.train_output = reshape_output(self.output_rows[:train_size], self.training_parameters.nb_classes)
            self.test_output = reshape_output(self.output_rows[train_size:], self.training_parameters.nb_classes)

    def prepare_for_training(self, model=None, training_parameters=None,
                             reshape_input=None, reshape_output=None,
                             prepare_model=None, **kwargs):
        """reshape unrefined data into a form that can be directly used by the model"""
        if model is None:
            if self.model is None:
                logging.error("model is not given for training")
                return
        else:
            self.model = model
        if training_parameters is None:
            if training_parameters is not None:
                self.training_parameters = training_parameters
        self.check_set_parameters()  # set default to any missing parameters
        if not hasattr(self, 'train_input'):
            self.split_dataset(reshape_input, reshape_output)
        if prepare_model is not None:
            prepare_model(self.model, **kwargs)  # compile the architecture if not compiled yet

    def train(self, **kwargs):
        if self.model is None:
            return None
        self.model.fit(self.train_input,
                       self.train_output,
                       batch_size=self.training_parameters.batch_size,
                       nb_epoch=self.training_parameters.nb_epoch,
                       show_accuracy=self.training_parameters.show_accuracy,
                       verbose=self.training_parameters.verbose,
                       validation_data=(self.test_input, self.test_output), **kwargs)
        return self.model.evaluate(self.test_input, self.test_output, show_accuracy=True, verbose=0)

    def handle_missing_data(self, number_handlers, string_handlers):
        """replace missing data using the handler functions
        
        @param number_handlers: replace NaN e.g. with average of all the values in the colum
        @param string_handlers: replace '' e.g. with the mode of all categories in the colum
        @todo refactor parameters, instead of two handlers, combine them, or get a boolean array
        to decide which one to apply"""
        if number_handlers is None or string_handlers is None:
            logging.warning("missing data handlers don't exist")
            return
        if hasattr(self, 'input_rows'):
            for i in range(0, len(self.input_rows[0])):
                # logging.warn(i)
                fill_missing_data(self.input_rows, i, number_handlers[i + self.output_count],
                                  string_handlers[i + self.output_count])
        if hasattr(self, 'output_rows'):
            if self.output_rows is not None and len(self.output_rows) > 0:
                for i in range(0, len(self.output_rows[0])):
                    fill_missing_data(self.output_rows, i, number_handlers[i], string_handlers[i])
        if hasattr(self, 'external_test_input'):
            if self.external_test_input is not None and len(self.external_test_input) > 0:
                for i in range(0, len(self.external_test_input[0])):
                    fill_missing_data(self.external_test_input, i, number_handlers[i + self.output_count],
                                      string_handlers[i + self.output_count])

    def _check_set_output_count(self):
        """try to autodetect from the labels if output count is set to <1"""
        if self.output_count < 1:
            logging.info("autodetecting output count from labels")
            self.output_count = 0
            for label in self.labels:
                if label[:5] == "label":
                    self.output_count += 1
            assert (self.output_count > 0)

    def load_train_test_csv(self, train_csv, test_csv, data_row=1, converters=None):
        """load data from csv train and test files
        
        @param data_row: the line number where real data starts,labels line is assumed to be given just before this line
        @param converters: a list of functions for each column to convert them to another format
        @param train_csv: train file that will be splitted into train and test data
        @param test_csv: file that contains the test values we want to use to make predictions
        """
        if train_csv is not None:
            labels, train_data = misc.load_csv(train_csv, data_row)
            self.labels = labels
            if self.supervised:
                self._check_set_output_count()
                assert (len(train_data) > 0)
                shuffle(train_data)  # so that the training does not depend on the order of training input
            if converters is None:
                logging.warning("applying no converter")
                (self.input_rows, self.output_rows) = misc.vertical_split(train_data, self.output_count)
            else:  # convert data too
                if len(converters) == 1:
                    logging.warning("applying only a single converter for training data")
                    converters *= len(train_data[0])
                self.input_rows = [
                    [converters[i + self.output_count](j) for (i, j) in enumerate(values[self.output_count:])]
                    for values in train_data]
                self.output_rows = [[converters[i](j) for (i, j) in enumerate(values[:self.output_count])]
                                    for values in train_data]
        if test_csv is not None:
            labels, test_data = misc.load_csv(test_csv, data_row)
            if not hasattr(self, 'labels'):
                self.labels = ['label'] * self.output_count + labels
            if converters is None:
                self.external_test_input = test_data
            else:
                if len(converters) == 1:
                    logging.warning("applying only a single converter for test data")
                    if train_csv is None:
                        converters = converters * len(test_data[0]) + self.output_count
                assert (len(converters) == len(
                        test_data[0]) + self.output_count)  # if not, test data doesnt match training data
                self.external_test_input = [[converters[i + self.output_count](j) for (i, j) in enumerate(values)]
                                            for values in test_data]

    def divide_data(self, train_data, test_data):
        """assumed that the data is already converted or it does not require conversion

        @param test_data: unrefined test data that will be used as external test input
        @param train_data: unrefined train data that will be vertically splitted into input and output rows
        """
        shuffle(train_data)  # so that the training does not depend on the order of training input
        self.input_rows = [values[self.output_count:]
                           for values in train_data]
        self.output_rows = [values[:self.output_count]
                            for values in train_data]
        assert (len(self.output_rows) == len(self.input_rows))
        self.external_test_input = test_data
        if len(self.external_test_input) > 0:
            assert (len(self.input_rows) > 0)
            logging.warning("external input %d" % len(self.external_test_input[0]))
            logging.warning("input %d" % len(self.input_rows[0]))
            assert (len(self.external_test_input[0]) == len(self.input_rows[0]))
        else:
            # logging.warning("test data is missing")
            pass

    """def divide_and_convert_data(self, train_data, test_data):
        shuffle(train_data)  # so that the training does not depend on the order of training input
        self.input_rows = [[self.converters[i + self.output_count](j) for (i, j) in enumerate(values[self.output_count:])] 
                        for values in train_data]
        self.output_rows = [[self.converters[i](j) for (i, j) in enumerate(values[:self.output_count])] 
                            for values in train_data]
        assert(len(self.output_rows) == len(self.input_rows))
        self.external_test_input = [[self.converters[i + self.output_count](j) for (i, j) in enumerate(values)] 
                        for values in test_data]
        if len(self.external_test_input) > 0:
            assert(len(self.input_rows) > 0)
            #logging.warning("external input %d" % len(self.external_test_input[0]))
            #logging.warning("input %d" % len(self.input_rows[0]))
            #logging.warning(self.input_rows[0])
            assert(len(self.external_test_input[0]) == len(self.input_rows[0]))
        else: 
            # logging.warning("test data is missing")
            pass"""

# a= Trainer("/home/agp/workspace/deep_learning/mnist/train.csv","/home/agp/workspace/deep_learning/mnist/test.csv")
# print(a.labels)
# print(a.input_rows[0])
# print(a.output_rows[0])
