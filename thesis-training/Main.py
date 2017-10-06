import sys

if sys.version_info[0] < 3:
    import got
else:
    import got3 as got
import train_database
import database
import time
from spider import *
from bson.objectid import ObjectId


def main():
    def printTweet(descr, t):
        print(descr)
        print("Username: %s" % t.username)
        print("Retweets: %d" % t.retweets)
        print("Text: %s" % t.text)
        print("Mentions: %s" % t.mentions)
        print("Hashtags: %s\n" % t.hashtags)

    # Example 1 - Get tweets by username
    tweetCriteria = got.manager.TweetCriteria().setUsername('barackobama').setMaxTweets(1)
    tweet = got.manager.TweetManager.getTweets(tweetCriteria)[0]

    printTweet("### Example 1 - Get tweets by username [barackobama]", tweet)

    # Example 2 - Get tweets by query search
    tweetCriteria = got.manager.TweetCriteria().setQuerySearch('europe refugees').setSince("2015-05-01").setUntil(
        "2015-09-30").setMaxTweets(1)
    tweet = got.manager.TweetManager.getTweets(tweetCriteria)[0]

    printTweet("### Example 2 - Get tweets by query search [europe refugees]", tweet)

    # Example 3 - Get tweets by username and bound dates
    tweetCriteria = got.manager.TweetCriteria().setUsername("barackobama").setSince("2015-09-10").setUntil(
        "2015-09-12").setMaxTweets(1)
    tweet = got.manager.TweetManager.getTweets(tweetCriteria)[0]

    printTweet("### Example 3 - Get tweets by username and bound dates [barackobama, '2015-09-10', '2015-09-12']",
               tweet)


def crawl():  # 1391037509
    # huff:  1360226790  ldschurch DynCorpIntl
    # bloom: 1388607922
    tweetCriteria = got.manager.TweetCriteria().setUsername("TMmeditation").setSince("2011-01-01").setUntil(
        "2017-09-30").setMaxTweets(5000)
    tweets = got.manager.TweetManager.getTweets(tweetCriteria)

    news_collection = train_database.train_collection()
    count = 0
    for tweet in tweets:
        count += 1
        urls = tweet.urls

        if 'twitter.com' in urls or not urls:
            continue

        if news_collection.find({'reference': urls}).count() > 0:
            print('Skip duplicated ' + urls)
            continue

        timestamp = int(time.mktime(time.strptime(tweet.formatted_date, '%a %b %d %H:%M:%S +0000 %Y')))
        document = {
            'id': tweet.id,
            'created_at': timestamp,
            'reference': tweet.urls
        }
        print('Insert ' + tweet.urls + '  created at ' + str(timestamp) + ' - ' + str(count))
        news_collection.insert_one(document)


def mergeDB():
    news_collection = database.news_collection()
    train_collection = train_database.train_collection()
    latest = ObjectId("5942946efe43ad1da80b1a79")
    count = 0
    index = 1

    while True:
        documents = news_collection.find({'_id': {'$gt': latest}}).limit(100)
        if documents.count() == 0:
            break
        for doc in documents:
            count += 1
            try:
                latest = doc['_id']
                if not doc.get('text'):
                    print('Skip', doc['reference'])
                    continue

                if train_collection.find({'reference': doc['reference']}).count() > 0:
                    print('Skip duplicated reference ' + doc['reference'])
                    continue
                if train_collection.find({'text': doc['text']}).count() > 0:
                    print('Skip duplicated text ' + doc['reference'])
                    continue

                document = {
                    'id': doc['id'],
                    'created_at': doc['created_at'],
                    'reference': doc['reference'],
                    'title': doc['title'],
                    'text': doc['text'],
                    'image': doc['image'],
                }
                print('Insert ' + doc['reference'] + '  created at ' + str(doc['created_at']))
                train_collection.insert_one(document)

            except Exception as e:
                print(e)


# CNBC business ABC TIME guardian HuffPostUK CNNent
# CNN ReutersWorld Forbes Telegraph SkyNews BBCNews guardiannews NBCNews HuffPost
if __name__ == '__main__':
     #crawl()
     locate_feeds(ObjectId("596352b38128941198844942"))
