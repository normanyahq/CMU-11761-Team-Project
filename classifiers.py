import pickle
from languagemodel import LanguageModel
import random
from utilities import logging
from sklearn import preprocessing
import numpy as np

class classify(LanguageModel):

    def save_model(self, model_file_name):
        '''
        Save your model.
        '''
        logging('Saving model into file: ' + model_file_name)
        pickle.dump(self.model, open(model_file_name, "w"))

    def load_model(self, model_file_name):
        '''
        Load your model
        '''
        logging('loading model from file: ' + model_file_name)

    def __init__(self, model_file_name=None,model=None):
        '''
        some initialization, do whatever you want
        like loading corpus, parameters, etc
        '''
        self.model = model
        self.model_obj = None
        self.model_file_name = model+ ".out"
       
    def train(self, docs, labels):
        '''
        docs: a list of docs, each doc is a list of sentences,
            '~~~~', '<s>' and '</s>' are removed
        labels: a integer list of labels,
            0: fake articles, 1: real articles
        '''
        # normalize features?
        #features = doc2feature(docs)
        '''
        newly added scaling
        '''
        features = preprocessing.scale(np.array([[1.0,1.0],[1.0,-1.0],[-1.0,1.0],[-1.0,-1.0]]))
        labels = np.array([1,1,0,0])
        logging('start training...')
        if self.model == 'logit': # logistic regression
            self.train_logit(features,labels)
        elif self.model == 'knn':
            self.train_knn(features,labels)
        elif self.model == 'svm':
            self.train_svm(features,labels)
        #elif self.model == 'adaboost':
        #    self.train_adaboost(features,labels)
        elif self.model == 'xgboost':
            self.train_xgboost(features,labels)
        else:
            print "No such model"
            raise
        logging('training with %s done ' %self.model)
        self.save_model(self.model+'.out')
        

    def predict(self, doc):
        '''
        for given string doc, return an integer either 0 or 1,
            0: fake article, 1: true article
        '''
        # try to use ensemble of different classifiers
        #features = doc2feature([doc])
        features = preprocessing.scale(np.array([[2.0,0.0],[-4,1]]))
        print "features",features
        print "mean",features.mean(axis=0)
        print "std",features.std(axis=0)
        predictions = self.model_obj.predict(features) # [[label]]
        probas = self.model_obj.predict_proba(features) # [[p0,p1]]
        print probas
        # probas for svm is different with training result, which is obtained by cv
        # it's also meaningless on small dataset
        print probas[0][0],probas[0][1],predictions[0]
        return

    def train_logit(self,features,labels):
        import sklearn.linear_model
        self.model_obj = sklearn.linear_model.LogisticRegression()
        self.model_obj.fit(features,labels)
    def train_knn(self,features,labels):
        import sklearn.neighbors
        self.model_obj = sklearn.neighbors.KNeighborsClassifier(n_neighbors=3) # avoid hard code later
        self.model_obj.fit(features,labels)
    def train_svm(self,features,labels):
        import sklearn.svm
        self.model_obj = sklearn.svm.SVC(probability=True)
        self.model_obj.fit(features,labels)
        #self.model_obj = SVC(features,labels)
    #def train_adaboost(features,labels):
    #    pass # try xgboost instead
    def train_xgboost(self,features,labels):
        import xgboost as xgb
        self.model_obj = xgb.XGBClassifier().fit(features,labels)
