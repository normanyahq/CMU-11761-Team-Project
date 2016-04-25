from __future__ import division
from utilities import *
import numpy as np
from multiprocessing import Pool
import pickle

def feature_avg_sentence_length(doc,parameter):
	return [sum([len(i) for i in doc])/len(doc)]
	
def calcu_blrt(pair,pair_dict,unigram_count,bigram_count,n1,n2):
	w1 = pair[0]
	w2 = pair[1]
	k1 = pair_dict[pair]
	k2 = bigram_count[pair]
	p1 = k1/n1
	p2 = k2/n2
	p = (k1+k2)/(n1+n2)
	#score =  2 * (np.log()+np.log()-np.log()-np.log()) # beaware of -inf
	score = k1 * np.log(p1) + (n1-k1) * np.log(1-p1)
	score += (k2 * np.log(p2) + (n2-k2) * np.log(1-p2))
	score -= (k1 * np.log(p) + (n1-k1) * np.log(1-p))
	score -= (k2 * np.log(p) + (n2-k2) * np.log(1-p))
	return score 

def feature_blrt(doc,doc_as_words):
	word_list = []
	for s in doc_as_words:
		word_list += s
	doc_as_words = word_list
	pair_dict = {}
	score_dict = {}
	w1 = None
	w2 = None
	for i in range(len(doc_as_words)):
		for j in range(len(doc_as_words)):
			if i == j:
				continue
			w1 = doc_as_words[i]
			w2 = doc_as_words[j]
			pair_dict[(w1,w2)] = pair_dict.get((w1,w2),0) + 1
	n1 = sum(pair_dict.values())
	n2 = sum(bigram_count.values())
	for i in range(len(doc_as_words)):
		for j in range(i+1,len(doc_as_words)):
			w1 = doc_as_words[i]
			w2 = doc_as_words[j]			
			score_dict[(w1,w2)] = calcu_blrt((w1,w2),pair_dict,unigram_count,bigram_count,n1,n2)
	return [sum(score_dict.values())/len(score_dict.keys()), max(score_dict.values()),min(score_dict.values())]

def skip_gram_perplexity(doc,model):
	return get_perplexity(doc,model)

def extract_feature(doc):
	res = []
	# feature 1: avg sentence length
	res.append(sentence_length(doc))
	# feature 2
	#skip_model = pickle.load('skip_gram.pkl')
	logging(str(res))
	return res

if __name__ == '__main__':
	docs, labels = load_data("data/train_text.txt", "data/train_label.txt")
	feature = []
	for doc in docs:
		feature.append(extract_feature(doc))

	pickle.dump(feature, open("feature_ang.pkl", "wb"))