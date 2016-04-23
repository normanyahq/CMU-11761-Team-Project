from __future__ import division
from utilities import *
import numpy as np
import pickle
import time
# unigram_log_prob = pickle.load(open("unigram_log_prob.pkl", "rb"))
# bigram_log_prob = pickle.load(open("bigram_log_prob.pkl", "rb"))
# trigram_log_prob = pickle.load(open("trigram_log_prob.pkl", "rb"))
# quadgram_log_prob = pickle.load(open("quadgram_log_prob.pkl", "rb"))
# log_prob_list = [quadgram_log_prob, trigram_log_prob, bigram_log_prob, unigram_log_prob]

def get_vocabulary(corpus_as_words):
    vocab = Set([])
    for doc_as_words in corpus_as_words:
        for words in doc_as_words:
            vocab.update(words)
    print "Vocab Size: %d"% len(vocab)
    return vocab

def get_unigram_count(corpus_as_words):
	unigram_count = {}
	for doc_as_words in corpus_as_words:
		for words in doc_as_words:
			for word in words:
				if word not in unigram_count:
					unigram_count[word] = 1
				else:
					unigram_count[word] += 1
	print "Unigram Count:", len(unigram_count)
	return unigram_count

def get_bigram_count(corpus_as_words):
	bigram_count = {}
	for doc_as_words in corpus_as_words:
		for words in doc_as_words:
			for i in range(1, len(words)):
				w1 = words[i-1]
				w2 = words[i]
				if (w1, w2) not in bigram_count:
					bigram_count[(w1, w2)] = 1
				else:
					bigram_count[(w1, w2)] += 1
	print "Bigram Count:", len(bigram_count)
	return bigram_count

def get_trigram_count(corpus_as_words):
	trigram_count = {}
	for doc_as_words in corpus_as_words:
		for words in doc_as_words:
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

def get_quadgram_count(corpus_as_words):
	quadgram_count = {}
	for doc_as_words in corpus_as_words:
		for words in doc_as_words:
			for i in range(3, len(words)):
				w1 = words[i-3]
				w2 = words[i-2]
				w3 = words[i-1]
				w4 = words[i]
				if (w1, w2, w3, w4) not in quadgram_count:
					quadgram_count[(w1, w2, w3, w4)] = 1
				else:
					quadgram_count[(w1, w2, w3, w4)] += 1
	print "Quadgram Count:", len(quadgram_count)
	return quadgram_count

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

def get_trigram_conditional_log_prob(trigram_count, bigram_count):
	trigram_prob = {}
	for w1, w2, w3 in trigram_count:
		trigram_prob[(w1, w2, w3)] = np.log2(trigram_count[(w1, w2, w3)]) - np.log2(bigram_count[(w1, w2)])
	print "calculated trigram conditional prob ... "
	return trigram_prob

def get_quadgram_conditional_log_prob(quadgram_count, trigram_count):
	quadgram_prob = {}
	for w1, w2, w3, w4 in quadgram_count:
		quadgram_prob[(w1, w2, w3, w4)] = np.log2(quadgram_count[(w1, w2, w3, w4)]) - np.log2(trigram_count[(w1, w2, w3)])
	print "calculated quadgram conditional prob ... "
	return quadgram_prob

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

def get_prob(log_prob_list, words, i):
	'''Back-off model, the last one must be unigram model with <UNK> vocab'''
	for log_prob_ind in range(len(log_prob_list)):
		log_prob = log_prob_list[log_prob_ind]
		element = iter(log_prob).next()
		if type(element) == str:
			k = words[i-1]
			# print 1
		else:
			hist_len = len(element)
			start = max(i-hist_len, 0)
			k = tuple(words[start:i])
			# print k, hist_len, start, i

		if k in log_prob:
			# print k
			return log_prob[k]

	unigram_prob = log_prob_list[-1]
	return unigram_prob.get('<UNK>', 1./len(unigram_prob))

def get_prob_skipgram(log_prob_list, words, i):
	'''Back-off model, the last one must be unigram model with <UNK> vocab'''
	if i > 0: 
		k = tuple(words[i-2:i+1])
		if k in log_prob_list[0]:
			print k
			return log_prob_list[0][k]

	for log_prob_ind in range(1, len(log_prob_list)):
		log_prob = log_prob_list[log_prob_ind]
		element = iter(log_prob).next()
		if type(element) == str:
			k = words[i-1]
			# print 1
		else:
			hist_len = len(element)
			start = max(i-hist_len, 0)
			k = tuple(words[start:i])
			# print k, hist_len, start, i

		if k in log_prob:
			# print k
			return log_prob[k]

	unigram_prob = log_prob_list[-1]
	return unigram_prob.get('<UNK>', 1./len(unigram_prob))

def get_log_likelihood(words, log_prob_list):
	lld = 0
	for i in range(1,len(words)+1):
		lld += get_prob(log_prob_list, words, i)
	return lld, len(words)

def get_sent_perplexity(words, log_prob_list):
	lld, word_num = get_log_likelihood(words, log_prob_list)
	lld /= word_num
	return 2**(-lld)

def get_doc_perplexity(doc_as_words, log_prob_list):
	lld = 0
	total_word = 0
	for words in doc_as_words:
		for i in range(1,len(words)+1):
			lld += get_prob(log_prob_list, words, i)
		total_word += len(words)
	lld /= total_word
	return 2**(-lld)

def get_doc_perplexity_skipgram(doc_as_words, log_prob_list):
	lld = 0
	total_word = 0
	for words in doc_as_words:
		for i in range(1,len(words)+1):
			lld += get_prob_skipgram(log_prob_list, words, i)
		total_word += len(words)
	lld /= total_word
	return 2**(-lld)


if __name__ == '__main__':
	docs, labels = load_data("data/train_text.txt", "data/train_label.txt")
	doc_as_words = get_docs_as_wordlist(docs)
	unigram_count = get_unigram_count(doc_as_words)
	bigram_count = get_bigram_count(doc_as_words)
	trigram_count = get_trigram_count(doc_as_words)
	quadgram_count = get_quadgram_count(doc_as_words)
	unigram_log_prob = get_unigram_log_prob(unigram_count)
	bigram_log_prob = get_bigram_conditional_log_prob(bigram_count, unigram_count)
	trigram_log_prob = get_trigram_conditional_log_prob(trigram_count, bigram_count)
	quadgram_log_prob = get_quadgram_conditional_log_prob(quadgram_count, trigram_count)
	skipgram_log_prob = get_skipgram_conditional_log_prob(trigram_count)

	sent = "YES THERE ARE ABOUT FOUR MINUTES PANELS ANIMALS"
	words = sent.split()
	print get_sent_perplexity(words, [quadgram_log_prob, skipgram_log_prob, trigram_log_prob, bigram_log_prob, unigram_log_prob])
