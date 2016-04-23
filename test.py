import sys
from classifiers import classify
c = classify(model=sys.argv[1])
c.train(1,1)
c.predict(1)
