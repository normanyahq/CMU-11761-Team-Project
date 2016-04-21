import pickle
from languagemodel import LanguageModel
import random
from utilities import logging


class RandomGuess(LanguageModel):

    def save_model(self, model_file_name):
        '''
        Save your model.
        '''
        logging('Saving model into file: ' + model_file_name)
        t_model = {'positive_rate': self.positive_rate}
        pickle.dump(t_model, open(model_file_name, "w"))

    def load_model(self, model_file_name):
        '''
        Load your model
        '''
        logging('loading model from file: ' + model_file_name)
        t_model = pickle.load(open(model_file_name))
        self.positive_rate = t_model['positive_rate']

    def __init__(self, model_file_name=None):
        '''
        some initialization, do whatever you want
        like loading corpus, parameters, etc
        '''
        if model_file_name:
            try:
                self.load_model(model_file_name)
            except IOError:
                logging('Model file not found, set positive_rate into 0.5')
                self.positive_rate = 0.5
        else:
            self.positive_rate = 0.5

    def train(self, docs, labels):
        '''
        docs: a list of docs, each doc is a list of sentences,
            '~~~~', '<s>' and '</s>' are removed
        labels: a integer list of labels,
            0: fake articles, 1: real articles
        '''
        logging('start training...')
        self.positive_rate = sum(labels) * 1. / len(labels)
        logging('training done, positive_rate: %f' % self.positive_rate)

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
        if self.positive_rate > 0.5:
            return self.positive_rate, 1 - self.positive_rate, 1
        else:
            return self.positive_rate, 1 - self.positive_rate, 0
