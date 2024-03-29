import pickle
import random
from utilities import logging


class LanguageModel():

    def save_model(self, model_file_name):
        '''
        Save your model.
        '''
        logging('Saving model into file: ' + model_file_name)

        # fill your model information here
        t_model = {}

        pickle.dump(t_model, open(model_file_name, "w"))

    def load_model(self, model_file_name):
        '''
        Load your model
        '''
        logging('loading model from file: ' + model_file_name)
        t_model = pickle.load(open(model_file_name))
        # copy your model information here

    def __init__(self, model_file_name=None):
        '''
        some initialization, do whatever you want
        like loading corpus, parameters, etc
        '''
        pass

    def train(self, docs, labels):
        '''
        docs: a list of docs, each doc is a list of sentences,
            '~~~~', '<s>' and '</s>' are removed
        labels: a integer list of labels,
            0: fake articles, 1: real articles
        '''
        logging('start training...')

        # implement your training algorithm here.

        logging('training done.')

    def predict(self, doc):
        '''
        for given string doc, return three values:
            P(fake|article), P(true|article) and class label:
            eg:
            0.791243 0.208757 0
            note that P(fake|article) + P(true|article) = 1
            and class label:
                0: True
                1: False
            the threshold for labeling true or false is not necessary to be 0.5
        '''
        pass
