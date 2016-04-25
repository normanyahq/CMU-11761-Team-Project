import pickle
from languagemodel import LanguageModel
import random
from utilities import logging
from sklearn import preprocessing
import numpy as np
from matplotlib import pyplot as plt
class classify(LanguageModel):

    def save_model(self, model_file_name):
        '''
        Save your model.
        '''
        #logging('Saving model into file: ' + model_file_name)
        pickle.dump(self.model_obj, open(model_file_name, "w"))

    def load_model(self, model_file_name):
        '''
        Load your model
        '''
        #logging('loading model from file: ' + model_file_name)

    def __init__(self, model_file_name=None,model=None):
        '''
        some initialization, do whatever you want
        like loading corpus, parameters, etc
        '''
        self.model = model
        self.model_obj = None
        self.model_file_name = './model/' + model+ ".out"
       
    def train(self, feature_pkl, label_pkl):
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
        #features = pickle.load(open('feature_perp_ratio_train.pkl'))
        features = pickle.load(open(feature_pkl))
        #features = [[x] for x in features] #!!!!!ONLY USED FOR SINGLE FEATURE
        labels = pickle.load(open(label_pkl))
        features = np.array(features)#preprocessing.scale(np.array(features))
        #print features[0]
        #labels = np.array(labels[int(len(labels)/10):])
        labels = np.array(labels)
        #print len(features),len(labels)
        #for i in range(len(features)):
        #    print "feature: %s,label: %d" %(features[i],labels[i])
        #labels = np.array([1,1,0,0])
        '''
        real_feature = []
        fake_feature = []
        for i in range(len(labels)):
            if labels[i] == 1:
                real_feature.append(features[i])
            else:
                fake_feature.append(features[i])
        plt.figure()
        bins = []
        step = 0
        for i in range(40):
            bins.append(step)
            step +=0.05
        plt.hist(real_feature,bins=bins)
        plt.hist(fake_feature,bins=bins)
        plt.title('Perplexity ratio of quadgram and trigram on truncated training set')
        plt.savefig('./report/FIG/030/ratio.png')
        '''
        #logging('start training...')
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
        #logging('training with %s done ' %self.model)
        self.save_model(self.model+'.out')
        

    def predict(self, feature_pkl):
        '''
        for given string doc, return an integer either 0 or 1,
            0: fake article, 1: true article
        '''
        # try to use ensemble of different classifiers
        #features = doc2feature([doc])
        #features = pickle.load(open('feature_perp_ratio_dev.pkl'))
        features = pickle.load(open(feature_pkl))
        labels = []
        #print "start predicting"
        for line in open('./data/dev_label.txt'):
            labels.append(int(line.strip()))
        #print features
        #labels = pickle.load(open('trun_label.pkl'))
        #labels = labels[int(len(labels)/10):2*int(len(labels)/10)]
        #labels = labels[:int(len(labels)/10)]
        labels = np.array(labels)
        #features = [[x] for x in features]
        features = np.array(features)
        #print features[0]
        predictions = self.model_obj.predict(features) # [[label]]
        probas = self.model_obj.predict_proba(features) # [[p0,p1]]
        for i in range(len(predictions)):
            print probas[i][0],probas[i][1],predictions[i]
        #print "sample prob",probas[0],labels[0]
        #for i in range(len(features)):
        #    print "feature: %f,label: %d,predictoin: %d" %(features[i][0],labels[i],predictions[i])
        '''
        real_feature = []
        fake_feature = []
        for i in range(len(labels)):
            #print labels[i], features[i][0]
            if labels[i] == 1:

                real_feature.append(features[i][0])
            else:
                fake_feature.append(features[i][0])
        
        plt.figure()
        plt.hist(real_feature)
        plt.hist(fake_feature)
        plt.savefig('pred_dist.png')
        #print probas
        # probas for svm is different with training result, which is obtained by cv
        # it's also meaningless on small dataset
        #print probas[0][0],probas[0][1],predictions[0]
        print predictions
        '''
        '''
        soft = 0.0
        for i in range(len(predictions)):
            soft += probas[i][labels[i]]
        soft /= len(predictions)
        #print "soft accuracy:", soft
        correct = 0
        for i in range(len(predictions)):
            if labels[i] == predictions[i]:
                correct += 1
        #print "accuracy:",correct * 1.0/len(predictions)
        '''
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
        #print features.size,labels.size
        self.model_obj = xgb.XGBClassifier().fit(features,labels)
