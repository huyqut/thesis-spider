from twitter_dev import TwitterDev
import re
import pymongo
import time


class Spider:

    @staticmethod
    def crawl_feeds(dev: TwitterDev = None):
        if dev is None:
            return
        mongo_client = pymongo.MongoClient('localhost', 27017)
        mongo_database = mongo_client['news-database']
        news_collection = mongo_database['news-collection']
        user_id = dev.api.VerifyCredentials().AsDict()['id']
        friends = dev.api.GetFriendIDs(user_id, stringify_ids = True)
        for status in dev.api.GetStreamFilter(follow = friends):
            message = status['text']
            url_match = re.search("(?P<url>https?://[^\s]+)", message)
            if url_match is None:
                continue
            url = url_match.group(0)
            if len(url) < 23:
                continue
            timestamp = time.strftime('%Y-%m-%d %H:%M:%S', time.strptime(status['created_at'],'%a %b %d %H:%M:%S +0000 %Y'))
            document = {
                'id': status['id'],
                'created_at': timestamp,
                'reference': url
            }
            news_collection.insert_one(document)
