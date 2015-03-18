import pickle
from nltk.tokenize import word_tokenize


def extract_features(document_words):
    # document_words = set(document)
    features = {}
    for word in word_features:
        features['contains(%s)' % word] = (word in document_words)
    return features

with open('./naive_classifier.bin', 'rb') as naive_classifier:
    classifier = pickle.load(naive_classifier)

with open('./word_features.bin', 'rb') as word_features_list:
    word_features = pickle.load(word_features_list)


tweet = "I don't like eating vegetable"

print classifier.classify(extract_features(word_tokenize(tweet)))

tweet = 'Your song is annoying'
print classifier.classify(extract_features(word_tokenize(tweet)))
