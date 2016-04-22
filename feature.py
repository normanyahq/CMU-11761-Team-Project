from __future__ import division
# from bllipparser import RerankingParser
# from nltk.parse.bllip import BllipParser
from utilities import *
from ngrammodel import *
import numpy as np
from multiprocessing import Pool
import pickle
import time
import gzip
# global log_prob_list

def get_parser_score(sent):
	try:
		best = rrp.parse(sent)[0]
		score = best.parser_score
		tree_len = len(best.ptb_parse.tokens())
		score /= tree_len
	except:
		score = 0

	return score

def feature_reranking_parser_score(doc, ifmean=True, ifmax=True, ifmin=True):
	# from nltk.data import find
	# model_dir = find('models/bllip_wsj_no_aux').path
	# rrp = BllipParser.from_unified_model_dir(model_dir)

	curTotal = 0
	curMax = -np.inf
	curMin = np.inf
	# print len(doc)

	# q = Queue()
	p_list = []

	# logging("Start parsing doc")

	
	scores = p.map(get_parser_score, doc)
	scores = [s for s in scores if s != 0]
	# logging(scores)

	res = []
	if ifmean:
		mean = np.sum(scores) / len(scores)
		res.append(mean)

	if ifmax:
		res.append(max(scores))

	if ifmin:
		res.append(min(scores))

	return res

def feature_quad_tri_perplexity_ratio(doc):
	t1 = time.time()
	quad_perplexity = get_doc_perplexity(doc, [quadgram_log_prob, trigram_log_prob, bigram_log_prob, unigram_log_prob])
	tri_perplexity = get_doc_perplexity(doc, [trigram_log_prob, bigram_log_prob, unigram_log_prob])
	t2 = time.time()

	print quad_perplexity, tri_perplexity, quad_perplexity/tri_perplexity
	return quad_perplexity/tri_perplexity


def extract_feature(doc):
	# res = feature_reranking_parser_score(doc)
	# res =
	# logging(str(res))
	# return res
	res = feature_quad_tri_perplexity_ratio(doc)
	return res

if __name__ == '__main__':
	# rrp = RerankingParser.fetch_and_load('WSJ-PTB3', verbose=True)
	p = Pool(10)

	docs, labels = load_data("data/train_text.txt", "data/train_label.txt")

	unigram_count = get_unigram_count(docs)
	bigram_count = get_bigram_count(docs)
	trigram_count = get_trigram_count(docs)
	quadgram_count = get_quadgram_count(docs)

	unigram_log_prob = get_unigram_log_prob(unigram_count)
	bigram_log_prob = get_bigram_conditional_log_prob(bigram_count, unigram_count)
	trigram_log_prob = get_trigram_conditional_log_prob(trigram_count, bigram_count)
	quadgram_log_prob = get_quadgram_conditional_log_prob(quadgram_count, trigram_count)

	# #global log_prob_list
	# log_prob_list = [quadgram_log_prob, trigram_log_prob, bigram_log_prob, unigram_log_prob]
	# pickle.dump(unigram_log_prob, gzip.open('unigram_log_prob.pkl.gz', 'wb'))
	# pickle.dump(bigram_log_prob, gzip.open('bigram_log_prob.pkl.gz', 'wb'))
	# pickle.dump(trigram_log_prob, gzip.open('trigram_log_prob.pkl.gz', 'wb'))
	# pickle.dump(quadgram_log_prob, gzip.open('quadgram_log_prob.pkl.gz', 'wb'))

	feature = []
	for doc in docs:
		# extract_feature(doc)
		feature.append(extract_feature(doc))

	# feature = p.map(extract_feature, docs)
	pickle.dump(feature, open("feature_perp_ratio.pkl", "wb"))