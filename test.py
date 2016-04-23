import sys
from classifiers import classify
c = classify(model=sys.argv[1])
c.train(sys.argv[2],sys.argv[4])
c.predict(sys.argv[3])
