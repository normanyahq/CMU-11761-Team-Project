from __future__ import division
from bllipparser import RerankingParser
# from nltk.parse.bllip import BllipParser
from utilities import *
import numpy as np


def rerankign_paser_score(doc, ifmean=True, ifmax=True, ifmin=True):
	rrp = RerankingParser.fetch_and_load('WSJ-PTB3', verbose=True)
	# from nltk.data import find
	# model_dir = find('models/bllip_wsj_no_aux').path
	# rrp = BllipParser.from_unified_model_dir(model_dir)

	curTotal = 0
	curMax = -np.inf
	curMin = np.inf
	# print len(doc)
	
	logging("Start parsing doc")
	for sent in doc:
		
		best = rrp.parse(sent)[0]
		score = best.parser_score
		tree_len = len(best.ptb_parse.tokens())
		score /= tree_len

		curTotal += score

		if score > curMax:
			curMax = score

		if score < curMin:
			curMin = score

	curMean = curTotal / len(doc)

	res = []
	if ifmean:
		res.append(curMean)
	if ifmax:
		res.append(curMax)
	if ifmin:
		res.append(curMin)

	return res



def extract_feature(doc):
	rerankign_paser_score(doc)

if __name__ == '__main__':
	docs, labels = load_data("data/train_text.txt", "data/train_label.txt")
	for doc in docs:
		rerankign_paser_score(doc)