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

	# print quad_perplexity, tri_perplexity, quad_perplexity/tri_perplexity
	return quad_perplexity/tri_perplexity

def get_content_words(unigram_count):
	sorted_x = sorted(unigram_count.items(), key=operator.itemgetter(1), reverse=True)
	content_words = [w for w, c in sorted_x[150:6500]]
	return content_words

def get_common_content_word_pairs(docs_as_words, labels, unigram_count, thresh=10):
	print "Get Real Docs ..."
	real_docs_as_words = get_real_docs(docs_as_words, labels)

	print "Real Doc Number:", len(real_docs_as_words)

	content_words = get_content_words(unigram_count)
	print "Content Word Number:", len(content_words)
	content_word_pair_count = get_content_word_pairs(real_docs_as_words, content_words)

	common_content_word_pair_count = {}
	for word_pair in content_word_pair_count:
		if content_word_pair_count[word_pair] > thresh:
			common_content_word_pair_count[word_pair] = content_word_pair_count[word_pair]
	#sorted_x = sorted(content_word_pair_count.items(), key=operator.itemgetter(1))
	print len(common_content_word_pair_count)
	return common_content_word_pair_count, content_words
	
def get_sent_common_content_word_pairs(words, common_content_word_pair_count, content_words, step=5):
	sent_len = len(words)
	ccw_cnt = 0
	for i in range(sent_len):
		w1 = words[i]
		if w1 in content_words:
			# print "in 1"
			for j in range(i+step, len(words)):
				w2 = words[j]
				if w2 in content_words and w1 != w2:
					# print "in 2"
					if (w1, w2) in common_content_word_pair_count or (w2, w1) in common_content_word_pair_count:
						# print "in 3"
						ccw_cnt += 1
	return ccw_cnt, sent_len



def feature_common_content_word_pairs(docs_as_words, common_content_word_pair_count, content_words):
	ccw_cnt_doc = 0
	total_sent_len = 0
	for doc_as_words in docs_as_words:
		for words in doc_as_words:
			ccw_cnt, sent_len = get_sent_common_content_word_pairs(words, common_content_word_pair_count, content_words)
			ccw_cnt_doc += ccw_cnt
			total_sent_len += sent_len
		print total_sent_len, ccw_cnt_doc


def get_content_word_pairs(docs_as_words, content_words, step=5):
	content_word_pair_count = {}
	it = 0
	for doc_as_words in docs_as_words:
		it += 1
		for words in doc_as_words:
			# appear at least step=5 words appart
			for i in range(len(words)):
				w1 = words[i]
				if w1 in content_words:
					# print "first word in"
					for j in range(i+step, len(words)):
						w2 = words[j]
						if w2 in content_words and w1 != w2:
							if (w1, w2) in content_word_pair_count:
								content_word_pair_count[(w1, w2)] += 1
							elif (w2, w1) in content_word_pair_count:
								content_word_pair_count[(w2, w1)] += 1
							else:
								content_word_pair_count[(w1, w2)] = 1
		if it % 50 == 0:
			print "Get content word pair: 50 Doc processed ..."

	print "Content Word Pair Count", len(content_word_pair_count)
	return content_word_pair_count


def extract_feature(doc, docs_as_words):
	# res = feature_reranking_parser_score(doc)
	# res =
	# logging(str(res))

	res = feature_quad_tri_perplexity_ratio(docs_as_words)
	return res

if __name__ == '__main__':
	# rrp = RerankingParser.fetch_and_load('WSJ-PTB3', verbose=True)
	p = Pool(10)

	docs, labels = load_data("data/train_text.txt", "data/train_label.txt")
	docs_as_words = get_docs_as_wordlist(docs)

	# unigram_count = get_unigram_count(docs_as_words)
	# common_content_word_pair_count, content_words = get_common_content_word_pairs(docs_as_words, labels, unigram_count)
	# pickle.dump(common_content_word_pair_count, open("common_content_word_pair_count.pkl", "wb"))
	# pickle.dump(content_words, open("content_words.pkl", "wb"))

	common_content_word_pair_count = pickle.load(open("common_content_word_pair_count.pkl", "rb"))
	content_words = pickle.load(open("content_words.pkl", "rb"))

	print common_content_word_pair_count
	feature_common_content_word_pairs(docs_as_words, common_content_word_pair_count, content_words)
	bigram_count = get_bigram_count(docs)
	# trigram_count = get_trigram_count(docs)
	# quadgram_count = get_quadgram_count(docs)

	# unigram_log_prob = get_unigram_log_prob(unigram_count)
	# bigram_log_prob = get_bigram_conditional_log_prob(bigram_count, unigram_count)
	# trigram_log_prob = get_trigram_conditional_log_prob(trigram_count, bigram_count)
	# quadgram_log_prob = get_quadgram_conditional_log_prob(quadgram_count, trigram_count)

	# feature = []
	# for doc in docs:
	# 	# extract_feature(doc)
	# 	feature.append(extract_feature(doc))

	# # feature = p.map(extract_feature, docs)
	# pickle.dump(feature, open("feature_perp_ratio.pkl", "wb"))