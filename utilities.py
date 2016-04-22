from parameters import *
import random
from sets import Set

def logging(message):
    if show_log:
        print message
    if save_log_to_file:
        with open(log_filename, "a") as f:
            f.write(str(message) + '\n')


def load_data(doc_file_name, label_file_name):
    '''
    Load docs and labels from files, and remove the '~~~~' and
        '<s>', '</s>' tags, re-organize them into lists:
            docs = [doc_0, doc_1, ... doc_n]
            doc_i = [sentence_0, sentence_1, ... sentence_n]
            sentence_i is a string
    return: docs: list of docs as described above,
            labels: list of integers, each is a label of corresponding doc
    '''
    docs = []
    labels = []
    with open(doc_file_name) as f:
        docs = f.read()
    with open(label_file_name) as f:
        labels = f.readlines()
    docs = docs.split('~~~~~')
    docs = [doc.replace('\n', '').replace('<s>', '').split('</s>') for doc in docs if doc]
    t_docs = []
    for doc in docs:
        doc = [sentence for sentence in doc if sentence.strip()]
        t_docs.append(doc)
    docs = t_docs

    labels = [int(i) for i in labels if i.strip().isdigit()]
    logging("Loaded %d docs, %d labels" % (len(docs), len(labels)))
    return docs, labels

def get_vocabulary(docs):
    vocab = Set([])
    for doc in docs:
        for sent in doc:
            words = sent.split()
            vocab.update(words)
    print "Vocab Size: %d"% len(vocab)
    return vocab


def cross_validation(model, docs, labels, batch_num):
    '''
    model: class instance inherited from languagemodel
    docs: list of documents, each document is a list of sentence
    labels: list of integers for labels
    batch_num: the number of group to divide the data into
    '''
    assert len(docs) == len(labels)
    positive_samples = [doc for i, doc in enumerate(docs) if labels[i]]
    negative_samples = [doc for i, doc in enumerate(docs) if not labels[i]]
    random.shuffle(positive_samples)
    random.shuffle(negative_samples)
    positive_per_batch = len(positive_samples) / batch_num
    negative_per_batch = len(negative_samples) / batch_num

    # accuracy
    average_accuracy = 0

    for i in range(0, batch_num):
        test_positive = positive_samples[
            i * positive_per_batch: min((i + 1) * positive_per_batch, len(positive_samples))]
        test_negative = negative_samples[
            i * negative_per_batch: min((i + 1) * negative_per_batch, len(negative_samples))]
        training_positive = positive_samples[:i * positive_per_batch] + \
            positive_samples[min((i + 1) * positive_per_batch, len(positive_samples)):]
        training_negative = negative_samples[:i * negative_per_batch] + \
            negative_samples[min((i + 1) * negative_per_batch, len(negative_samples)):]
        logging('Training batch %d...' % (i + 1))
        model.train(training_positive + training_negative,
                    [1] * len(training_positive) + [0] * len(training_negative))
        logging('Traning done, start predicting...')
        # True Positive, False Positive, True Negative, False Negative
        tp = fp = tn = fn = 0.
        for doc in test_positive:
            if model.predict(doc):
                tp += 1
            else:
                fn += 1
        for doc in test_negative:
            if model.predict(doc):
                fp += 1
            else:
                tn += 1
        testset_size = len(test_positive) + len(test_negative)
        accuracy = (tp + tn) * 1. / testset_size
        logging("Test Doc: %d\ntp=%d(%.2f%%), fp=%d(%.2f%%), tn=%d(%.2f%%), fn=%d(%.2f%%)\nAccuracy=%.2f\n"
                % (testset_size, tp, tp * 100. / len(test_positive), fp, fp * 100. / len(test_positive),
                   tn, tn * 100. / len(test_negative), fn, fn * 100. / len(test_negative),
                   accuracy))
        average_accuracy += accuracy
    average_accuracy /= batch_num
    logging("Average Accuracy: %.2f%%" % (average_accuracy * 100))

if __name__ == '__main__':
    docs, labels = load_data("data/train_text.txt", "data/train_label.txt")
    get_vocabulary(docs)
