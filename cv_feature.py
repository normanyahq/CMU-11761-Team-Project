import sys
import pickle 
from sklearn import cross_validation
import sklearn.linear_model
# load features
features = pickle.load(open(sys.argv[1]))
#features = [[x] for x in features]
labels = pickle.load(open(sys.argv[2]))
obj = sklearn.linear_model.LogisticRegression()
print 'start cv...'
scores = cross_validation.cross_val_score(obj,features,labels,cv=10)
print scores