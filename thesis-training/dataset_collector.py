import sys

if sys.version_info[0] < 3:
    import got
else:
    import got3 as got
import database
import time

# CNBC business ABC TIME guardian HuffPostUK CNNent CNN
def getData():
    tweetCriteria = got.manager.TweetCriteria().setUsername("CNN").setSince("2013-01-01").setUntil(
        "2017-01-01").setMaxTweets(15000)
    tweets = got.manager.TweetManager.getTweets(tweetCriteria)
    
    news_collection = database.news_collection()
    
    for tweet in tweets:
    
        urls = tweet.urls
    
        if 'twitter' in urls or not urls:
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
        print('Insert ' + tweet.urls + '  created at ' + str(timestamp))
        news_collection.insert_one(document)