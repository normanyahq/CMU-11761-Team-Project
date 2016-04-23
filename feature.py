from __future__ import division
#from bllipparser import RerankingParser
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
import argparse
from parameters import *

def get_parser_score(sent):
	try:
		best = rrp.parse(sent)[0]
		score = best.parser_score
		tree_len = len(best.ptb_parse.tokens())
		score /= tree_len
	except:
		score = 0

	return score

def feature_reranking_parser_score(doc, doc_as_words):
	curTotal = 0
	curMax = -np.inf
	curMin = np.inf

	p_list = []

	scores = p.map(get_parser_score, doc)
	scores = [s for s in scores if s != 0]

	res = []
	if fparse_mean:
		mean = np.sum(scores) / len(scores)
		res.append(mean)

	if fparse_max:
		res.append(max(scores))

	if fparse_min:
		res.append(min(scores))

	return res

def feature_quad_tri_perplexity_ratio(doc, doc_as_words):
	t1 = time.time()
	quad_perplexity = get_doc_perplexity(doc_as_words, [quadgram_log_prob, trigram_log_prob, bigram_log_prob, unigram_log_prob])
	tri_perplexity = get_doc_perplexity(doc_as_words, [trigram_log_prob, bigram_log_prob, unigram_log_prob])
	t2 = time.time()

	return [quad_perplexity/tri_perplexity]

def get_content_stop_words(unigram_count, cw_start=150, cw_end=6500, sw_end=50):
	sorted_x = sorted(unigram_count.items(), key=operator.itemgetter(1), reverse=True)
	content_words = [w for w, c in sorted_x[cw_start:cw_end]]
	stop_words = [w for w, c in sorted_x[:sw_end]]
	print "Content Stop Words Done ..."
	return Set(content_words), Set(stop_words)

def get_common_content_word_pairs(docs, real_docs_as_words, content_words, thresh=90, load=False):
	if load:
		cw_cnt_dict = pickle.load(open("cw_cnt_dict.pkl", "rb"))
	else:
		print "get_common_content_word_pairs ..."	
		cw_cnt_dict = {}
		cw_instance_cnt = 0
		for docid in range(len(docs)):
			doc = docs[docid]
			
			[pair_corr_list, pair_corr_list_5, word_dict] = generate_pairs(doc)
			for pair in pair_corr_list_5:
				words = pair.split()
				if words[0] in content_words and words[1] in content_words:
					cw_instance_cnt += 1
					if pair in cw_cnt_dict:
						cw_cnt_dict[pair] += 1
					else:
						cw_cnt_dict[pair] = 1

			if docid % 200 == 199:
				print "doc", docid
		pickle.dump(cw_cnt_dict, open("cw_cnt_dict.pkl", "wb"))

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
	
def feature_common_content_word_pairs(doc, doc_as_words):
	ccw_cnt  = 0.0
	# print le n(doc)
	[pair_corr_list, pair_corr_list_5, word_dict] = generate_pairs(doc)

	pair_cnt = sum([len(pair_corr_list[p]) for p in pair_corr_list ])
	if pair_cnt < 1:
		print "Too short doc ..."
		return [0]

	# print "generate_pairs done ..."
	for pair in pair_corr_list_5:
		if pair in ccw_list:
			ccw_cnt += len(pair_corr_list_5[pair])
			# print len(pair_corr_list_5[pair])
	return [1.0 * ccw_cnt/pair_cnt]

def feature_content_and_stopwords(doc, doc_as_words):
	# the percentage of the doc that are content_words or stop_words:
	cw_cnt = 0.0
	sw_cnt = 0.0
	total_words = 0.0

	# longest concecutive stop words / content words
	prev_word = 0 # 0 Neither;  1 stopword;  2 content word
	longest_ccw = 0
	longest_csw = 0
	
	for words in doc_as_words:
		total_words += len(words)
		cur_ccw = 1
		cur_csw = 1
		prev_word = 0
		for word in words:
			if word in stop_words:
				sw_cnt += 1
				if prev_word == 1:
					cur_csw += 1 
				prev_word = 1
			elif word in content_words:
				cw_cnt += 1

				if prev_word == 2:
					cur_ccw += 1
				prev_word = 2
			else:
				if cur_ccw > longest_ccw:
					longest_ccw = cur_ccw
				if cur_csw > longest_csw:
					longest_csw = cur_csw
				cur_ccw = 1
				cur_csw = 1
				prev_word = 0

	return [cw_cnt/total_words, sw_cnt/total_words, longest_ccw, longest_csw]

def extract_feature(ffunc_list, doc, doc_as_words):
	# res = feature_reranking_parser_score(doc)
	# res =
	# logging(str(res))

	# res = feature_quad_tri_perplexity_ratio(docs_as_words)


	# res = feature_common_content_word_pairs(doc, ccw_list)
	res = []
	tmp = None
	for fn in ffunc_list:
		tmp = fn(doc, doc_as_words)
		if type(tmp) is list:
			if type(tmp[0]) in (list,tuple):
				for e in tmp:
					for ee in e:
						res.append(ee)
			else:
				res += tmp
		else:
			res.append(tmp)
	# logging(str(res))
	return res

##### set dependency #####
# pay attention to the order please
dependency = {}
dependency["fparser"] = Set(["rrp"])
dependency["f43pprt"] = Set(["unigram", "bigram", "trigram", "quadgram"])
dependency["fccwp"] = Set(["cs_words", "ccw_list"])
dependency["fcs"] = Set(["cs_words"])

def isdefined(param_name):
	res = param_name in locals() or param_name in globals()
	# print param_name, res
	return res

if __name__ == '__main__':
	p = Pool(10)
	ffunc_list = []
	dep = Set([])
	parser = argparse.ArgumentParser()
	parser.add_argument("-td", "--train_doc", type=str, help="set training document path (pickle only)")
	parser.add_argument("-tl", "--train_label", type=str, help="set training label path (pickle only)")

	parser.add_argument("-dd", "--dev_doc", type=str, help="set dev document path (pickle only)")
	parser.add_argument("-dl", "--dev_label", type=str, help="set dev label path (pickle only)")

	parser.add_argument("-pd", "--pred_doc", type=str, help="set prediction document path (pickle only)")

	parser.add_argument("-fparser", help="add feature_reranking_parser_score", action="store_true")
	parser.add_argument("-f43pprt", help="add feature_quad_tri_perplexity_ratio", action="store_true")
	parser.add_argument("-fccwp", help="add feature_common_content_word_pairs", action="store_true")
	parser.add_argument("-fcs", help="add feature_content_and_stopwords", action="store_true")
	'''
		Added features from yanran
	'''
	parser.add_argument("-fss", help="add feature_simple_statistics", action="store_true")
	parser.add_argument("-fpc", help="add feature_percentage_corr", action="store_true")
	parser.add_argument("-fup", help="add feature_unseen_pairs", action="store_true")
	parser.add_argument("-fr", help="add feature_repetition", action="store_true")
	parser.add_argument("-frcs",help="add feature_ratio_content_stop", action="store_true")
	parser.add_argument("-fcscore",help="add feature_coherence_score", action="store_true")

	'''
		Added features from anglu
	'''

	parser.add_argument("-fs", "--fsave", type=str, help="save feature to pickle")

	args = parser.parse_args()
	if args.train_doc and args.train_label:
		train_docs = pickle.load(open(args.train_doc, "rb"))
		train_labels =  pickle.load(open(args.train_label, "rb"))

		train_docs_as_words = get_docs_as_wordlist(train_docs)
		real_docs_as_words = get_real_docs(train_docs_as_words, train_labels)

	if args.dev_doc and args.dev_label:
		dev_docs = pickle.load(open(args.dev_doc, "rb"))
		dev_labels =  pickle.load(open(args.dev_label, "rb"))

		dev_docs_as_words = get_docs_as_wordlist(dev_docs)

	if args.pred_doc:
		pred_docs = pickle.load(open(args.pred_doc, "rb"))
		pred_docs_as_words = get_docs_as_wordlist(pred_docs)

	if args.fparser:
		ffunc_list.append(feature_reranking_parser_score)
		dep.update(dependency["fparser"])
	if args.f43pprt:
		ffunc_list.append(feature_quad_tri_perplexity_ratio)
		dep.update(dependency["f43pprt"])
	if args.fccwp:
		ffunc_list.append(feature_common_content_word_pairs)
		dep.update(dependency["fccwp"])
	if args.fcs:
		ffunc_list.append(feature_content_and_stopwords)
		dep.update(dependency["fcs"])
	
	if args.fss:
		ffunc_list.append(feature_simple_statistics)
		#dep.update(dependency["fss"])
	if args.fpc:
		ffunc_list.append(feature_percentage_corr)
		#dep.update(dependency["fpc"])
	if args.fup:
		ffunc_list.append(feature_unseen_pairs)
		#dep.update(dependency["fup"])
	if args.fup:
		ffunc_list.append(feature_repetition)
		#dep.update(dependency["fr"])	
	if args.fup:
		ffunc_list.append(feature_ratio_content_stop)
		#dep.update(dependency["frcs"])
	if args.fup:
		ffunc_list.append(feature_coherence_score)
		#dep.update(dependency["fcscore"])
	
	print "ffunc_list",ffunc_list

	for d in dep:
		if d == "rrp":
			rrp = RerankingParser.fetch_and_load('WSJ-PTB3', verbose=True)
		if d == "unigram":
			if not isdefined("unigram_count"):
				unigram_count = get_unigram_count(train_docs_as_words)
			unigram_log_prob = get_unigram_log_prob(unigram_count)
		if d == "bigram":
			if not isdefined("bigram_count"):
				bigram_count = get_bigram_count(train_docs_as_words)
			if not isdefined("unigram_count"):
				unigram_count = get_unigram_count(train_docs_as_words)
			bigram_log_prob = get_bigram_conditional_log_prob(bigram_count, unigram_count)
		if d == "trigram":
			if not isdefined("trigram_count"):
				trigram_count = get_trigram_count(train_docs_as_words)
			if not isdefined("bigram_count"):
				bigram_count = get_bigram_count(train_docs_as_words)
			trigram_log_prob = get_trigram_conditional_log_prob(trigram_count, bigram_count)
		if d == "quadgram":
			if not isdefined("quadgram_count"):
				quadgram_count = get_quadgram_count(train_docs_as_words)
			if not isdefined("trigram_count"):
				trigram_count = get_trigram_count(train_docs_as_words)
			quadgram_log_prob = get_quadgram_conditional_log_prob(quadgram_count, trigram_count)
		if d == "cs_words":
			if not isdefined("unigram_count"):
				unigram_count = get_unigram_count(train_docs_as_words)
			content_words, stop_words = get_content_stop_words(unigram_count)
		if d == "ccw_list":
			# if not isdefined("content_words"):
			# 	if isdefined(unigram_count):
			# 		unigram_count = get_unigram_count(train_docs_as_words)
			# 	content_words, stop_words = get_content_stop_words(unigram_count)
			ccw_list = get_common_content_word_pairs(train_docs, real_docs_as_words, content_words, ccw_list_thresh, ccw_list_load)

	
	feature = []

	for docid, doc, doc_as_words, label in zip(range(len(train_docs)), train_docs, train_docs_as_words, train_labels):
		ft = extract_feature(ffunc_list, doc, doc_as_words)
		#print ft, label
		feature.append(ft)

		if docid % 200 == 199:
			print ft, label

		if args.fsave:
			pickle.dump(feature, open("train_"+args.fsave + ".pkl", "wb"))

	dev_feature = []
	if isdefined("dev_docs"):
		for docid, doc, doc_as_words, label in zip(range(len(dev_docs)), dev_docs, dev_docs_as_words, dev_labels):
			ft = extract_feature(ffunc_list, doc, doc_as_words)
			# print ft, label
			dev_feature.append(ft)

			if docid % 200 == 199:
				print ft, label

		if args.fsave:
			pickle.dump(dev_feature, open("dev_"+args.fsave + ".pkl", "wb"))

	if isdefined("pred_docs"):
		for docid, doc, doc_as_words in zip(range(len(pred_docs)), pred_docs, pred_docs_as_words):
			ft = extract_feature(ffunc_list, doc, doc_as_words)
			# print ft, label
			dev_feature.append(ft)

			if docid % 200 == 199:
				print ft, label

		if args.fsave:
			pickle.dump(dev_feature, open("pred_"+args.fsave, "wb"))






