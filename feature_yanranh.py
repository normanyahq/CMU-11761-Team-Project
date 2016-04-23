import utilities
from collections import defaultdict
import numpy

global pair_corr_score
global pair_corr_score_5
global pair_corr_list
global pair_corr_list_5
global word_dict
global unseen_pairs

# functions to call to generate features:
# 1. feature_simple_statistics(doc):
#    return: 12 features in two list(each list have 6 features)
#    this is the function to generate simple statistics about correlation values
#
# 2. feature_percentage_corr:
#    return: 1 feature
#    this is the function to percentage of correlation values above some threshold
#
# 3. feature_unseen_pairs:
#    return: 2 feature in one list [num of unseen_pairs, percent of unseen pairs]
#    this is the function to generate the number of unseen pairs and the percent of unseen pairs


# this is the function to generate Simple Statistics features
def feature_simple_statistics(doc):
    global pair_corr_score
    global pair_corr_score_5

    if len(pair_corr_score) == 0:
        get_corr(doc)

    # doc is an individual article
    # Simple Statistics(mean, median, maximum, minimum, range and variance) or word pair correlation values
    # Statistic considering all word pairs & only pairs have distance at least 5 words

    mean = numpy.mean(pair_corr_score)
    median = numpy.median(pair_corr_score)
    var = numpy.std(pair_corr_score)
    max = numpy.max(pair_corr_score)
    min = numpy.min(pair_corr_score)
    range = max - min

    feature_simple = [mean, median, var, max, min, range]

    mean_5 = numpy.mean(pair_corr_score_5)
    median_5 = numpy.median(pair_corr_score_5)
    var_5 = numpy.std(pair_corr_score_5)
    max_5 = numpy.max(pair_corr_score_5)
    min_5 = numpy.min(pair_corr_score_5)
    range_5 = max_5 - min_5

    feature_simple_5 = [mean_5, median_5, var_5, max_5, min_5, range_5]

    return [feature_simple, feature_simple_5]

# this is the function to generate word pairs in a doc
# return 1. all word pairs 2. words pairs have distance >= 5
def generate_pairs(doc):
    global pair_corr_list
    global pair_corr_list_5
    global word_dict
    global unseen_pairs

    unseen_pairs = 0
    pair_corr_list = defaultdict(list())
    pair_corr_list_5 = defaultdict(list())
    # list = (sid1, sid2, ..., sidn)

    word_dict = defaultdict(list())
    # word_dict[word] is the occurence list of word in the whole doc

    # generate pairs list
    for sid in range(len(doc)):
        sentence = doc[sid]
        for i in range(len(sentence)):
            word = sentence[i]
            if word not in word_dict:
                word_dict[word] = [sid]
            else:
                word_dict[word].append(sid)

            for j in range(i + 1, len(sentence)):
                pair = [sentence[i], sentence[j]]

                if pair not in pair_corr_list:
                    pair_corr_list[pair] = [sid]
                    unseen_pairs += 1
                else:
                    pair_corr_list[pair].append(sid)
                if j - i >= 5:
                    if pair not in pair_corr_list_5:
                        pair_corr_list_5[pair] = [sid]
                    else:
                        pair_corr_list_5[pair].append(sid)

    return [pair_corr_list, pair_corr_list_5]

# this is the funciton to get the correlation value of word pairs
# return two score lists
def get_corr(doc):
    global pair_corr_score
    global pair_corr_score_5
    global pair_corr_list
    global pair_corr_list_5
    global word_dict

    pair_corr_score = list()
    pair_corr_score_5 = list()

    if len(pair_corr_list) == 0:
        generate_pairs(doc)

    # traverse pair_corr_list to calculate Q statistics
    for pair in pair_corr_list:
        pair_matrix = pair_corr_list[pair]
        c11 = len(set(pair_matrix))
        c12 = len(set(word_dict[pair[0]])) - c11
        c21 = len(set(word_dict[pair[1]])) - c11
        c22 = len(doc) - c11 - c12 - c21
        q = (c11 * c22 - c12 * c21) / (c11 * c22 + c12 * c21)
        pair_corr_score.append(q)

    for pair in pair_corr_list_5:
        pair_matrix = pair_corr_list_5[pair]
        c11 = len(set(pair_matrix))
        c12 = len(set(word_dict[pair[0]])) - c11
        c21 = len(set(word_dict[pair[1]])) - c11
        c22 = len(doc) - c11 - c12 - c21
        q = (c11 * c22 - c12 * c21) / (c11 * c22 + c12 * c21)
        pair_corr_score_5.append(q)

# this is to generate the feature of percentage of correlation values above a threshold
def feature_pencentage_corr(doc, threshold):
    global pair_corr_score
    # global pair_corr_score_5

    if len(pair_corr_score) == 0:
        get_corr(doc)

    count = 0
    for q in pair_corr_score:
        if q >= threshold:
            count += 1

    return float(count) / len(pair_corr_score)


def feature_percentage_corr(doc):
    threshold = 0.3
    return feature_pencentage_corr(doc, threshold)


# this is to generate the number of unseen pairs and the percent of unseen pairs
# return [num of unseen_pairs, percent of unseen pairs]
def feature_unseen_pairs(doc):
    global unseen_pairs
    global pair_corr_list

    if len(pair_corr_list) == 0:
        generate_pairs(doc)

    pairs_sum = 0
    for sentence in doc:
        pairs_sum += len(sentence) * (len(sentence) - 1) / 2

    return [unseen_pairs, float(unseen_pairs) / pairs_sum]


def get_stop_word_list():
    data_file = './data/stop_word_list.txt'
    data = open(data_file, 'r')

    stop_words = list()

    for line in data:
        word = line.replace("\n", "")
        stop_words.append(word)

    return stop_words


