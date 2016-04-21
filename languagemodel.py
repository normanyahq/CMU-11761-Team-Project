import pickle


class LanguageModel:

    def save_model(self):
        '''
        Save your model.
        '''
        pass

    def load_model(self):
        '''
        Load your model
        '''
        pass

    def __init__(self, model_file_name=None):
        '''
        some initialization, do whatever you want
        like loading corpus, parameters, etc
        '''
        # define the filename of model to load
        self._model_file_name = model_file_name
        pass

    def train(self, docs, labels):
        '''
        sentence: a string list of docs
        labels: a integer list of labels,
            0: fake articles, 1: real articles
        '''
        pass

    def predict(self, doc):
        '''
        for given string doc, return an integer either 0 or 1,
            0: fake article, 1: true article
        '''
        pass
