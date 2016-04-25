import sys
import pickle
from classifiers import classify
from matplotlib import pyplot as plt
def plot_trunc_len_dist(trun_doc_pkl,title,save_path):

    ls = []
    docs = pickle.load(open(trun_doc_pkl))
    for doc in docs:
        ls.append(len(doc))
    plt.hist(ls)
    plt.title(title)
    plt.savefig(save_path)

def plot_feature_hist(features,labels,feature_name):
    OUTDIR = './test/plot/'
    #features = pickle.load(open(train_feature_pkl))
    #labels = pickle.load(open(label_pkl))
    if len(labels) != len(features):
        print "Feature & label length don't match!"
        return
    if type(features[0]) in (list,tuple):
        print "Can't plot dist for list/tuple feature"
        return
    real_feature = []
    fake_feature = []
    for i in range(len(label)):
        if labels[i] == 1:
            real_feature.append(features[i])
        elif labels[i] == 0:
            fake_feature.append(features[i])
        else:
            print "Unseen label"
            return 0
    plt.features
    plt.hist(real_feature)
    plt.hist(fake_feature)
    plt.label(feature_name)
    plt.savefig(OUTDIR+feature_name+'.png')

if __name__ == '__main__':
    c = classify(model=sys.argv[1])
    c.train(sys.argv[2],sys.argv[4])
    c.predict(sys.argv[3])

