from __future__ import division
from bllipparser import RerankingParser
# from nltk.parse.bllip import BllipParser
from utilities import *
import numpy as np
from multiprocessing import Pool
import pickle

rrp = RerankingParser.fetch_and_load('WSJ-PTB3', verbose=True)

def get_parser_score(sent):
	best = rrp.parse(sent)[0]
	score = best.parser_score
	tree_len = len(best.ptb_parse.tokens())
	score /= tree_len

	return score

def reranking_parser_score(doc, ifmean=True, ifmax=True, ifmin=True):
	# from nltk.data import find
	# model_dir = find('models/bllip_wsj_no_aux').path
	# rrp = BllipParser.from_unified_model_dir(model_dir)

	curTotal = 0
	curMax = -np.inf
	curMin = np.inf
	# print len(doc)

	# q = Queue()
	p_list = []

	logging("Start parsing doc")

	p = Pool(10)
	scores = p.map(get_parser_score, doc)
	logging(scores)

	res = []
	if ifmean:
		mean = np.sum(scores) / len(doc)
		res.append(mean)

	if ifmax:
		res.append(max(scores))

	if ifmin:
		res.append(min(scores))

	return res



def extract_feature(doc):
	res = reranking_parser_score(doc)
	logging(str(res))
	return res

if __name__ == '__main__':
	docs, labels = load_data("data/train_text.txt", "data/train_label.txt")
	feature = []
	for doc in docs:
		feature.append(extract_feature(doc))

	pickle.dump(feature, open("feature.pkl", "wb"))