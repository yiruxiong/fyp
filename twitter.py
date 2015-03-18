# coding=utf-8

import sys
reload(sys)
sys.setdefaultencoding("utf-8")

import tweepy
import re
import MySQLdb as mdb
from _mysql_exceptions import OperationalError

# this is how to define a function
# def <function name>:
#     indent at least one space, normally we do 4 spaces
#     all indents under same level must be same
def processTweet(text):
    if re.match(r'^RT', text):
        # None is a Python type, just like null in Java
        return None, None
    re_hashtag = re.search(r'#(\w+)', text)

    # if we find hashtag then extract it
    # else we set hashtag as NULL in database
    # for if-else branch, we write
    # if <cond>:
    #    indent
    # elif <cond>:
    #    indent
    # else:
    #    indent
    if re_hashtag:
        hashtag = re_hashtag.group(0)
    else:
        hashtag = 'NULL'

    # please read damn long regular expression defination below
    # encode utf-8 means convert the whole string into utf-8 format
    # please read text format here http://baike.baidu.com/view/1204863.htm
    # emoji, such as 😄 is under utf-8 code
    #text = re.sub(r'(https?:\/\/[\S]+)|#(\w+)|(@ ?\w+)|^ +', '', text, flags=re.MULTILINE).encode('utf-8')
    text = re.sub(r'(https?:\/\/[\S]+)|^ +', '', text, flags=re.MULTILINE).encode('utf-8')
    # in python, we can return multiple values together as a tuple
    return text, hashtag


def getNewTweets():
    # we plan to get 100000 tweets in total
    max_tweets = 500    # the number of tweets we want to collect each time
    chunk_size = 10

    last_id = -1
    # count how many tweets we got in total
    count = 0
    #loop until the number reached we acquired
    while count < max_tweets:
        try:
            # everytime, we get 10 tweets from server
            new_tweets = api.search(q=query, count=chunk_size, max_id=str(last_id - 1))
            # no tweet exists
            if not new_tweets:
                break

            # loop inside 10 tweets we just got
            for tweet in new_tweets:
                time = tweet.created_at
                username = tweet.user.name
                # this is how to receive values
                # you can either use 
                #   (text, hashtag) = processTweet(tweet.text)
                # or
                #   text, hashtag = processTweet(tweet.text)
                # to receive function's returned value
                text, hashtag = processTweet(tweet.text)
                # if no text at all
                if text is None:
                    # directly go to next loop
                    continue
                # this is how to insert, but this is only remin in sql's cache
                try:
                    cur.execute("INSERT INTO tweets(Text, Hashtag, Time, Username) VALUES(%s, %s, %s, %s)", (text,hashtag,time,username))
                except OperationalError:
                    print "skip 1"
                    continue
            # commit is important! it means write queries inside memory into database
            con.commit()
            # accumulate count
            count += chunk_size
            # [-1] means that the last item inside the list
            # such as a = [1,2,3,4] then a[-1] = 4
            last_id = new_tweets[-1].id
            print "Add %d tweets to database, %d tweets remain" %(count, max_tweets - count)

        # Python as means give a alian name for something
        except tweepy.TweepError as e:
            # better print the error while developing
            print e
            # break is break the loop
            break

# this is the main entrance to run a Python program, formal
# just like the main function in Java
if __name__ == '__main__':
    consumer_key = "aCPgiKAunVppzeyUizvgvzi3f"
    consumer_secret = "UsqU5kh9UQMiTME4gYIe87QIr0KsKwGqTI3Td61qdY0q2oiMU3"
    access_key = "2314012285-WXZu5ZnqFMFGKLSmeSEUNVTacqSWAAKdbjw5Mef"
    access_secret = "HY2mQ2dzQFW1yeVl0dyyo94tDTGxYeFWNsRCakthRzJpS"
    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_key, access_secret)
    api = tweepy.API(auth)

    sql_user = 'tweet'
    sql_passwd = 'NtzGPN8HwBnd63QE'
    sql_name = 'tweet_sent'
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

    public_tweets = api.home_timeline()
    query = 'tesco'
      
    getNewTweets()

# # this is how to query a table
# cur.execute('SELECT * from tweets')
# for row in cur:
#     # row[1:] means everything in row except the first one, make them as a new list
#     # *row[1:] means unpack the list
#     # There's also star in function definition, which means "all other positional arguments", 
#     # you can read this http://stackoverflow.com/questions/36901/what-does-double-star-and-star-do-for-python-parameters
#     print u'{} {}'.format(*row[1:]).encode('utf-8')


# # everytime terminate database connection is a good habit
# con.close()
