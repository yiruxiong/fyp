__author__ = 'yiruxiong'

# coding=utf-8

import sys
reload(sys)
sys.setdefaultencoding("utf-8")

import nltk
import numpy
import pickle
from nltk.tokenize import word_tokenize
from sklearn import cross_validation
from nltk.classify import apply_features

def extract_features(document_words):
    # document_words = set(document)
    features = {}
    for word in word_features:
        features['contains(%s)' % word] = (word in document_words)
    return features

if __name__ == '__main__':

    with open('./word_features.bin', 'rb') as word_features_list:
        word_features = pickle.load(word_features_list)

    with open('./tweets_list.bin', 'rb') as tweets_list:
        tweets = pickle.load(tweets_list)

    training_set = apply_features(extract_features, tweets)

    target = numpy.array([sentiment for text, sentiment in training_set])
    train = numpy.array([text for text, sentiment in training_set])


    #Simple K-Fold cross validation. 10 folds.
    cv = cross_validation.KFold(len(training_set), n_folds=10, shuffle=False, random_state=None)

    #iterate through the training and test cross validation segments and
    #run the classifier on each one, aggregating the results into a list
    results_naive = []
    classifier_naive_best = None
    acc_classifier_naive_best = 0
    for traincv, testcv in cv:
        classifier_naive = nltk.NaiveBayesClassifier.train(zip(train[traincv], target[traincv]))
        classifier_naive.show_most_informative_features(10)
        acc_classifier_naive = nltk.classify.accuracy(classifier_naive, zip(train[testcv], target[testcv]))
        print acc_classifier_naive
        results_naive.append(acc_classifier_naive)
        if acc_classifier_naive > acc_classifier_naive_best:
            classifier_naive_best = classifier_naive
            acc_classifier_naive_best = acc_classifier_naive
    assert classifier_naive_best is not None

    #print out the mean of the cross-validated results
    print "Results_naive: " + str(numpy.array(results_naive).mean())

    # save the classifier
    with open('./naive_classifier.bin', 'wb') as naive_classifier:
        pickle.dump(classifier_naive_best, naive_classifier)