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

        logging('training done, positive_rate: %f' % self.positive_rate)

    def predict(self, doc):
        '''
        for given string doc, return an integer either 0 or 1,
            0: fake article, 1: true article
        '''
        pass
