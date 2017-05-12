from twitter_dev import TwitterDev
import re
import pymongo
import time
import geograpy
import threading


class Spider:

    def __init__(self, dev: TwitterDev = None):
        self.dev = dev


    def crawl_feeds(self):
        if self.dev is None:
            return
        mongo_client = pymongo.MongoClient('localhost', 27017)
        mongo_database = mongo_client['news-database']
        news_collection = mongo_database['news-collection']
        user_id = self.dev.api.VerifyCredentials().AsDict()['id']
        friends = self.dev.api.GetFriendIDs(user_id, stringify_ids = True)
        for status in self.dev.api.GetStreamFilter(follow = friends):
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

    def locate_feeds(self):

        def parse_pages(url, collection):
            try:
                geo = geograpy.get_place_context(url)
                if len(geo.places) > 0:
                    for place in geo.places:
                        result = collection.find_one({'place': place })
                        if result is None:
                            collection.insert_one({'place': place,
                                                   'count': 1})
                        else:
                            result['count'] = result['count'] + 1
                            collection.replace_one({'_id': result['_id']}, result)
            except Exception as e:
                pass

        mongo_client = pymongo.MongoClient('localhost', 27017)
        mongo_database = mongo_client['news-database']
        news_collection = mongo_database['news-collection']
        documents = news_collection.find({})
        location_collection = mongo_database['location-collection']
        tasks = []
        for doc in documents:
            try:
                ref = doc['reference']
                thread = threading.Thread(target = parse_pages, args = (ref, location_collection))
                tasks.append(thread)
                thread.start()
                if len(tasks) == 10:
                    for task in tasks:
                        task.join()

            except Exception as e:
                pass

