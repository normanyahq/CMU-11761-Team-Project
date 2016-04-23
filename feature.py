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
import operator
from feature_yanranh import *
from collections import defaultdict
from sets import Set

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

def feature_quad_tri_perplexity_ratio(docs_as_words):
	t1 = time.time()
	quad_perplexity = get_doc_perplexity(docs_as_words, [quadgram_log_prob, trigram_log_prob, bigram_log_prob, unigram_log_prob])
	tri_perplexity = get_doc_perplexity(docs_as_words, [trigram_log_prob, bigram_log_prob, unigram_log_prob])
	t2 = time.time()

	return quad_perplexity/tri_perplexity

def get_content_words(unigram_count):
	sorted_x = sorted(unigram_count.items(), key=operator.itemgetter(1), reverse=True)
	content_words = [w for w, c in sorted_x[150:6500]]
	print "Content Words Done ..."
	return Set(content_words)

def get_common_content_word_pairs(docs, real_docs_as_words, content_words, thresh=90):
	# print "get_common_content_word_pairs ..."	
	# cw_cnt_dict = {}
	# cw_instance_cnt = 0
	# for docid in range(len(docs)):
	# 	doc = docs[docid]
		
	# 	[pair_corr_list, pair_corr_list_5, word_dict] = generate_pairs(doc)
	# 	for pair in pair_corr_list_5:
	# 		words = pair.split()
	# 		if words[0] in content_words and words[1] in content_words:
	# 			cw_instance_cnt += 1
	# 			if pair in cw_cnt_dict:
	# 				cw_cnt_dict[pair] += 1
	# 			else:
	# 				cw_cnt_dict[pair] = 1

	# 	if docid % 200 == 199:
	# 		print "doc", docid
	# pickle.dump(cw_cnt_dict, open("cw_cnt_dict.pkl", "wb"))

	cw_cnt_dict = pickle.load(open("cw_cnt_dict.pkl", "rb"))
	# sorted_x = sorted(cw_cnt_dict.items(), key=operator.itemgetter(1))
	# for x in sorted_x:
	# 	print x

	cw_cnt_list = np.array(cw_cnt_dict.values())
	thresh_val = np.percentile( cw_cnt_list,thresh)
	print "common_content_word_pairs Using Thresh Val:", thresh_val
	ccw_cnt_dict = {}
	for pair in cw_cnt_dict:
		if cw_cnt_dict[pair] > thresh_val:
			ccw_cnt_dict[pair] = cw_cnt_dict[pair]

	return Set(ccw_cnt_dict.keys())
	
def feature_common_content_word_pairs(doc, ccw_list):
	ccw_cnt  = 0.0
	# print le n(doc)
	[pair_corr_list, pair_corr_list_5, word_dict] = generate_pairs(doc)

	pair_cnt = sum([len(pair_corr_list[p]) for p in pair_corr_list ])
	if pair_cnt < 1:
		print "Too short doc ..."
		return 0

	# print "generate_pairs done ..."
	for pair in pair_corr_list_5:
		if pair in ccw_list:
			ccw_cnt += len(pair_corr_list_5[pair])
			# print len(pair_corr_list_5[pair])
	return 1.0 * ccw_cnt/pair_cnt

def extract_feature(doc, docs_as_words):
	# res = feature_reranking_parser_score(doc)
	# res =
	# logging(str(res))

	# res = feature_quad_tri_perplexity_ratio(docs_as_words)


	res = feature_common_content_word_pairs(doc, ccw_list)
	return res

if __name__ == '__main__':
	# rrp = RerankingParser.fetch_and_load('WSJ-PTB3', verbose=True)
	p = Pool(10)

	#docs, labels = load_data("data/train_text.txt", "data/train_label.txt")
	docs = pickle.load(open('trun_doc.pkl'))
	labels = pickle.load(open('trun_label.pkl'))
	docs_as_words = get_docs_as_wordlist(docs)
	real_docs_as_words = get_real_docs(docs_as_words, labels)

	unigram_count = get_unigram_count(docs_as_words)
	# bigram_count = get_bigram_count(docs_as_words)
	# trigram_count = get_trigram_count(docs_as_words)
	# quadgram_count = get_quadgram_count(docs_as_words)

	# unigram_log_prob = get_unigram_log_prob(unigram_count)
	# bigram_log_prob = get_bigram_conditional_log_prob(bigram_count, unigram_count)
	# trigram_log_prob = get_trigram_conditional_log_prob(trigram_count, bigram_count)
	# quadgram_log_prob = get_quadgram_conditional_log_prob(quadgram_count, trigram_count)
	content_words = get_content_words(unigram_count)
	ccw_list = get_common_content_word_pairs(docs, real_docs_as_words, content_words, 99)
	# pickle.dump(ccw_list, open("ccw_list.pkl","wb"))
	# ccw_list = pickle.load(open("ccw_list.pkl","rb"))
	feature = []

	for docid, doc, doc_as_words, label in zip(range(len(docs)), docs, docs_as_words, labels):
		ft = extract_feature(doc, doc_as_words)
		# print ft, label
		feature.append(ft)

		if docid % 200 == 199:
			print ft, label

	# # feature = p.map(extract_feature, docs)
	pickle.dump(feature, open("feature_common_content_pair.pkl", "wb"))