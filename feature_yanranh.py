import utilities
from collections import defaultdict
import numpy
import pickle

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
#
# 4. feature_repetition:
#    return: 2 features in one list [percent of repetition, the length of the longest repeated phrase]
# 5. def feature_ratio_content_stop:
#    return: 1 feature
#    this is the ratio of content words and stop words (ratio_content_stop = #stop_words / #content_words)
#

# this is the function to generate Simple Statistics features
def feature_simple_statistics(doc):
    # global pair_corr_score
    # global pair_corr_score_5

    # if pair_corr_score is None:
    [pair_corr_score, pair_corr_score_5] = get_corr(doc)

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

    if len(pair_corr_score_5) == 0:
        mean_5 = 0.0
        median_5 = 0.0
        var_5 = 0.0
        max_5 = 0.0
        min_5 = 0.0
    else:
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
    # global pair_corr_list
    # global pair_corr_list_5
    # global word_dict
    global unseen_pairs

    unseen_pairs = 0
    pair_corr_list = defaultdict(list)
    pair_corr_list_5 = defaultdict(list)
    # list = (sid1, sid2, ..., sidn)

    word_dict = defaultdict(list)
    # word_dict[word] is the occurence list of word in the whole doc

    # generate pairs list
    for sid in range(len(doc)):
        sentence = doc[sid].split(' ')
        for i in range(len(sentence)):
            word = sentence[i]
            if word == ' ' or word == '':
                continue

            if word not in word_dict:
                word_dict[word] = [sid]
            else:
                word_dict[word].append(sid)

            for j in range(i + 1, len(sentence)):
                if sentence[j] == ' ' or sentence[j] == '' or sentence[j] == sentence[i]:
                    continue
                if sentence[i] <= sentence[j]:
                    pair = str(sentence[i]) + ' ' + str(sentence[j])
                else:
                    pair = str(sentence[j]) + ' ' + str(sentence[i])

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

    return [pair_corr_list, pair_corr_list_5, word_dict]


# this is the funciton to get the correlation value of word pairs
# return two score lists
def get_corr(doc):
    # global pair_corr_score
    # global pair_corr_score_5
    # global pair_corr_list
    # global pair_corr_list_5
    # global word_dict

    pair_corr_score = list()
    pair_corr_score_5 = list()

    # if pair_corr_list is None:
    [pair_corr_list, pair_corr_list_5, word_dict] = generate_pairs(doc)

    # traverse pair_corr_list to calculate Q statistics
    for pair in pair_corr_list:
        pair_matrix = pair_corr_list[pair]
        c11 = len(set(pair_matrix))
        c12 = len(set(word_dict[pair.split(' ')[0]])) - c11
        c21 = len(set(word_dict[pair.split(' ')[1]])) - c11
        c22 = len(doc) - c11 - c12 - c21
        q = float(c11 * c22 - c12 * c21) / (c11 * c22 + c12 * c21)
        # if q < -1:
        #     print "whoops"
        pair_corr_score.append(q)

    for pair in pair_corr_list_5:
        pair_matrix = pair_corr_list_5[pair]
        c11 = len(set(pair_matrix))
        c12 = len(set(word_dict[pair.split(' ')[0]])) - c11
        c21 = len(set(word_dict[pair.split(' ')[1]])) - c11
        c22 = len(doc) - c11 - c12 - c21
        q = float(c11 * c22 - c12 * c21) / (c11 * c22 + c12 * c21)
        # if q < -1:
        #     print "whoops"
        pair_corr_score_5.append(q)

    return [pair_corr_score, pair_corr_score_5]

# this is to generate the feature of percentage of correlation values above a threshold
def feature_pencentage_corr(doc, threshold):
    global pair_corr_score
    # global pair_corr_score_5

    # if pair_corr_score is None:
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

    # if pair_corr_list is None:
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
        word = line.replace("\n", "").upper()
        stop_words.append(word)

    return stop_words


# return: 2 features in one list [percent of repetition, the length of the longest repeated phrase]
# note: single word are also included as general "phrases"
def feature_repetition(doc):
    phrase_list = list()
    repetition_count = 0
    max_phrase_length = 0
    word_length_sum = 0

    for sentence in doc:
        sentence = sentence.split(' ')
        word_length_sum += len(sentence)
        for i in range(len(sentence)):
            phrase = ''
            for j in range(i, len(sentence)):
                phrase += ' ' + sentence[j]
                if phrase in phrase_list:
                    repetition_count += 1
                    # update the length of longest repeated phrase
                    if len(phrase) > max_phrase_length:
                        max_phrase_length = len(phrase)
                else:
                    phrase_list.append(phrase)

    return [float(repetition_count) / word_length_sum, max_phrase_length]


#    return: 1 feature
#    this is the ratio of content words and stop words (ratio_content_stop = #stop_words / #content_words)
def feature_ratio_stop_content(doc):
    stop_words = get_stop_word_list()
    content_words = pickle.load(open("content_words.pkl", "rb"))
    stop_words_count = 0
    content_words_count = 0


    for sentence in doc:
        sentence = sentence.split()
        for word in sentence:
            if word in stop_words:
                stop_words_count += 1
            if word in content_words:
                content_words_count += 1

    ratio_stop_content = float(stop_words_count) / content_words_count
    return ratio_stop_content



def main():
    docs, labels = utilities.load_data('./data/dev_text.txt', './data/dev_label.txt')
    last = len(docs) - 1
    for i in range(len(docs)):
        print i
        print feature_ratio_stop_content(docs[i])
        # print feature_simple_statistics(docs[i])
        # print feature_unseen_pairs(docs[i])
    #     print feature_repetition(docs[i])
    #     print feature_unseen_pairs(docs[i])
    # print feature_repetition(docs[199])
    # print feature_simple_statistics(docs[62])
    # print feature_simple_statistics(docs[last])


if __name__ == "__main__":
    main()


