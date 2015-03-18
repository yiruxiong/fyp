__author__ = 'yiruxiong'

# coding=utf-8

import sys
reload(sys)
sys.setdefaultencoding("utf-8")

import nltk
import re
import pickle
import MySQLdb as mdb
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer


def processTweet(text):
    re_hashtag = re.search(r'#(\w+)', text)
    # if we find hashtag then extract it
    # else we set hashtag as NULL in database
    if re_hashtag:
        hashtag = re_hashtag.group(0)
    else:
        hashtag = 'NULL'

    # encode utf-8 means convert the whole string into utf-8 format
    # please read text format here http://baike.baidu.com/view/1204863.htm
    # emoji, such as is under utf-8 code
    #text = re.sub(r'(https?:\/\/[\S]+)|#(\w+)|(@ ?\w+)|^ +', '', text, flags=re.MULTILINE).encode('utf-8')

    text = re.sub(r'(https?:\/\/\S+)', '', text, flags=re.MULTILINE).encode('utf-8')
    # in python, we can return multiple values together as a tuple
    return text


def removeRedudant(word):
    pattern = r'(\w)\1{2,}'
    return ''.join([w for w in re.split(pattern, word) if w is not ''])


def negate_features(text):
    def add_neg(t):
        return t.group(1) + " " + " ".join(["NOT_%s" % i for i in t.group(2).split(" ") if len(i) > 3]) + t.group(3)

    # exp = r"(don'?t|doesn'?t|didn'?t|not)([a-z ]*)([,.])"   #72.2
    exp = r"(don'?t|doesn'?t|didn'?t|not|n't)([a-z ]*)([,.]?)"  #71.8

    # exp = r"(n'?t|not)([a-z ]*)([,.])"   #70.5

    return re.sub(exp, add_neg, text.lower())

    # a = 'llllllooooovvvvvveeeeeaaafdd'


def wordnet_lemmarizer(text, postag):
    wordnet_lemmatizer = WordNetLemmatizer()

    if postag in ["VB", "VBD", "VBN", "VBZ", "VBG", "VBP"]:
        wordnet_twe = wordnet_lemmatizer.lemmatize(text, pos='v')
    else:
        wordnet_twe = wordnet_lemmatizer.lemmatize(text, pos='n')
    return wordnet_twe


def get_words_in_tweets(tweets):
    all_words = []
    for (words, _) in tweets:
        all_words.extend(words)
    return all_words


def get_word_features(wordlist):
    wordlist = nltk.FreqDist(wordlist)
    word_features = wordlist.keys()
    return word_features


if __name__ == '__main__':
    sql_user = 'root'
    sql_passwd = '123456'
    sql_name = 'training_data'
    # set up sql connection
    # this is one way to determ python function arguments
    # please read http://www.tutorialspoint.com/python/python_functions.htm
    con = mdb.connect(host='localhost',
                      user=sql_user,
                      passwd=sql_passwd,
                      db=sql_name,
                      charset='utf8mb4',
                      use_unicode=True)
    cur = con.cursor()

    count = 0
    labeled_tweets = []

    try:
        cur.execute("SELECT `Text` , `Sentiment` FROM `training` ORDER BY rand(100) LIMIT 100")
        row = cur.fetchone()
        while row is not None:
            text = row[0]
            sentiment = row[1]
            labeled_tweets.append((text, sentiment))

            if (count % 10 == 0):
                print "Extracted %d items from database" % count
            count += 1
            row = cur.fetchone()
    except:
        print "Error: unable to fetch data"

    # disconnect from server
    con.close()

    tweets = []
    text_list = []
    sentiments = []
    word_tokenize_list = []
    word_pos = []
    test = []
    for (words, sentiment) in labeled_tweets:
        # remove URL
        prepro_twe = processTweet(words)

        # remove repeat letters
        process_twe = removeRedudant(prepro_twe)

        # negate the sentence for words not dont doesnt
        neg_words = negate_features(process_twe)

        try:
            # use nltk tokenize the words
            word_tokenize_list = word_tokenize(neg_words)
        except UnicodeDecodeError:
            print "'utf8' codec can't decode byte 0xc2 in position 2: unexpected end of data"

        words_filtered = []
        for word_token in word_tokenize_list:
            if len(word_token) >= 2:
                words_filtered.append(word_token)

        # use part-of-speech tagging
        word_pos_list = nltk.pos_tag(words_filtered)
        sentiments.append(sentiment)

        wordnet_twes = []
        for text, pos in word_pos_list:
            wordnet_twe = wordnet_lemmarizer(text, pos)
            wordnet_twes.append(wordnet_twe)
        tweets.append((wordnet_twes,sentiment))
        word_pos.append(word_pos_list)
        test.append(wordnet_twes)

    word_features = get_word_features(get_words_in_tweets(tweets))

    with open('./word_features.bin', 'wb') as word_features_list:
        pickle.dump(word_features, word_features_list)

    with open('./tweets.bin', 'wb') as tweets_list:
        pickle.dump(tweets, tweets_list)