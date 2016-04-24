'''
This module is used to truncate the docs in training set,
according to the distribution of number of sentences in development set.
http://www.cs.cmu.edu/~roni/11761/project.html

num_sent  weight
1           20
2           10
3           10
4           10
5           10
7           10
10          10
15          10
20          10
'''


import random

# sentence length = 1 has double weight than others, so
# here's a trick for random.choice
num_sentence_list = [1, 1, 2, 3, 4, 5, 7, 10, 15, 20]


def truncate_docs(docs, labels):
    '''
    input:
        docs: a list of docs, each doc is a list of sentences
        labels: a list of integers, either 0 or 1
    output:
        r_docs: a list of docs, each doc has a max sentence length according to the setting
        r_labels: a list of corresponding labels to r_docs
    '''
    r_docs = []
    r_labels = []
    for i, doc in enumerate(docs):
        if len(docs) <= max(num_sentence_list):
            r_docs.append(doc)
            r_labels.append(labels[i])
            continue
        pos = 0
        while pos < len(doc):
            t_doc_length = random.choice(num_sentence_list)
            r_docs.append(doc[pos:min(pos + t_doc_length, len(doc))])
            r_labels.append(labels[i])
            pos += t_doc_length

    return r_docs, r_labels
