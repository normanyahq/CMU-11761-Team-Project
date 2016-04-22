from __future__ import division
from utilities import *
import numpy as np

def get_vocabulary(corpus):
    vocab = Set([])
    for doc in corpus:
        for sent in doc:
            words = sent.split()
            vocab.update(words)
    print "Vocab Size: %d"% len(vocab)
    return vocab

def get_unigram_count(corpus):
	unigram_count = {}
	for doc in corpus:
		for sent in doc:
			words = sent.split()
			for word in words:
				if word not in unigram_count:
					unigram_count[word] = 1
				else:
					unigram_count[word] += 1
	print "Unigram Count:", len(unigram_count)
	return unigram_count

def get_bigram_count(corpus):
	bigram_count = {}
	for doc in corpus:
		for sent in doc:
			words = sent.split()
			for i in range(1, len(words)):
				w1 = words[i-1]
				w2 = words[i]
				if (w1, w2) not in bigram_count:
					bigram_count[(w1, w2)] = 1
				else:
					bigram_count[(w1, w2)] += 1
	print "Bigram Count:", len(bigram_count)
	return bigram_count

def get_trigram_count(corpus):
	trigram_count = {}
	for doc in corpus:
		for sent in doc:
			words = sent.split()
			for i in range(2, len(words)):
				w1 = words[i-2]
				w2 = words[i-1]
				w3 = words[i]
				if (w1, w2, w3) not in trigram_count:
					trigram_count[(w1, w2, w3)] = 1
				else:
					trigram_count[(w1, w2, w3)] += 1
	print "Trigram Count:", len(trigram_count)
	return trigram_count

def count_dict_val(dic):
	total = 0
	for k in dic:
		total += dic[k]
	return total

def get_unigram_log_prob(unigram_count):
	total_count = count_dict_val(unigram_count)
	unigram_prob = {}
	for k in unigram_count:
		unigram_prob[k] = np.log2(unigram_count[k]) - np.log2(total_count)
	print "calculated unigram prob ... "
	return unigram_prob

def get_bigram_conditional_log_prob(bigram_count, unigram_count):
	bigram_prob = {}
	for w1, w2 in bigram_count:
		bigram_prob[(w1, w2)] = np.log2(bigram_count[(w1, w2)]) - np.log2(unigram_count[w1])
	print "calculated bigram conditional prob ... "
	return bigram_prob

def get_skipgram_conditional_log_prob(trigram_count):
	skipgram_count = {}
	for w1, w2, w3 in trigram_count:
		if (w1, w3) in skipgram_count:
			skipgram_count[(w1, w3)] += trigram_count[(w1, w2, w3)]
		else:
			skipgram_count[(w1, w3)] = trigram_count[(w1, w2, w3)]

	skipgram_prob = {}
	for (w1, w2, w3) in trigram_count:
		skipgram_prob[(w1, w2, w3)] = np.log2(trigram_count[(w1, w2, w3)]) - np.log2(skipgram_count[(w1, w3)])
	print "calculated skipgram conditional prob ... "
	return skipgram_prob

def get_prob(log_prob_list, word):
	'''Back-off model, the last one must be unigram model with <UNK> vocab'''
	for log_prob in log_prob_list:
		if word in log_prob:
			return log_prob[word]
	return log_prob_list[-1]['<UNK>']

def get_log_likelihood(sent, log_prob_list):
	lld = 0
	words = sent.split()
	for w in words:
		lld += get_prob(log_prob_list, w)
	return lld / len(words)

def get_perplexity(sent, log_prob_list):
	lld = get_log_likelihood(sent, log_prob_list)
	return 2**(-lld)

if __name__ == '__main__':
	docs, labels = load_data("data/train_text.txt", "data/train_label.txt")
	unigram_count = get_unigram_count(docs)
	bigram_count = get_bigram_count(docs)
	trigram_count = get_trigram_count(docs)
	unigram_log_prob = get_unigram_log_prob(unigram_count)
	bigram_log_prob = get_bigram_conditional_log_prob(bigram_count, unigram_count)
	skipgram_log_prob = get_skipgram_conditional_log_prob(trigram_count)

	sent = "I HAVE A DREAM"
	print get_perplexity(sent, [skipgram_log_prob, bigram_log_prob, unigram_log_prob])
