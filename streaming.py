
from sListener import SListener
import time, tweepy, sys

## authentication
consumer_key = "aCPgiKAunVppzeyUizvgvzi3f"
consumer_secret = "UsqU5kh9UQMiTME4gYIe87QIr0KsKwGqTI3Td61qdY0q2oiMU3"
access_key = "2314012285-WXZu5ZnqFMFGKLSmeSEUNVTacqSWAAKdbjw5Mef"
access_secret = "HY2mQ2dzQFW1yeVl0dyyo94tDTGxYeFWNsRCakthRzJpS"
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_key, access_secret)
api = tweepy.API(auth)



def main():
    track = ['obama', 'romney']

    listen = SListener(api, 'myprefix')
    stream = tweepy.Stream(auth, listen)

    print "Streaming started..."

    try:
        stream.filter(track = track)
    except:
        print "error!"
        stream.disconnect()

if __name__ == '__main__':
    main()