import re
import time
import geograpy
import train_database
import thesis_logging
import numpy as np
from threading import Thread
from newspaper import Article
from builtins import any
from bson.objectid import ObjectId

crawler_finish = False


def locate_feeds(latest=ObjectId("5950c589f296532a806e4f31")):
    global crawler_finish
    logger = thesis_logging.get_logger('locator')
    news_collection = train_database.train_collection()

    class PageParser(Thread):
        def __init__(self, tweet_id, url):
            super().__init__()
            self.tweet_id = tweet_id
            self.url = url

        def run(self):
            try:
                print('Parse ' + self.url)
                article = Article(self.url)
                article.download()

                # if article.download_exception_msg and "404" in article.download_exception_msg:
                #     logger.error('404 not found, delete... ' + self.url)
                #     news_collection.remove({"id": self.tweet_id})
                #     return
                # if article.download_exception_msg and "410" in article.download_exception_msg:
                #     logger.error('410 client error, delete... ' + self.url)
                #     news_collection.remove({"id": self.tweet_id})
                #     return
                article.parse()
                ignore_list = ["twitter.com", "youtube.com", "facebook.com", "instagram.com"]
                if any(x in article.canonical_link for x in ignore_list):
                    print('delete ' + article.canonical_link)
                    news_collection.remove({"id": self.tweet_id})
                    return

                print(
                    'Title for ' + article.top_image + '  -  ' + article.canonical_link + '\n' + article.title + '\n\n')
                print('Latest: ' + str(latest))

                if news_collection.find({'$or': [{'title': article.title}, {'text': article.text}]}).count() > 0:
                    print('Duplicate, Ignore!')
                    news_collection.remove({"id": self.tweet_id})
                    return

                vector = 0
                news_collection.update_one({'id': self.tweet_id},
                                           {'$set': {
                                                     'vector': vector,
                                                     'title': article.title,
                                                     'text': article.text,
                                                     'image': article.top_image}})
            except Exception as e:
                logger.error(str(e))

    tasks = []
    while True:
        print('Start Locating')
        # documents = news_collection.find({'_id': {'$gt': latest}}).limit(100)
        documents = news_collection.aggregate(
            [{'$match': {'text': {'$exists': False}}}, {'$sample': {'size': 100}}])
        # print('Found ' + str(documents.count()) + ' after ' + str(latest))

        # Clean up remaining tasks
        if len(tasks) != 0:
            print('Cleaning up remaining tasks')
            for task in tasks:
                task.join()
            tasks.clear()

        # if documents.count() == 0:
        #    break

        index = 0

        for doc in documents:
            try:
                ref = doc['reference']
                latest = doc['_id']
                image = doc.get('image')
                if image is not None:
                    print('image skip')
                    continue
                if news_collection.find({'reference': ref}).count() > 1:
                    print('delete duplicate ' + ref)
                    news_collection.remove({"id": doc['id']})
                    continue

                thread = PageParser(doc['id'], ref)
                tasks.append(thread)
                thread.start()
                time.sleep(8)
                index += 1
                if index % 5 == 0:
                    logger.info('Start to wait')
                    for task in tasks:
                        task.join()
                    logger.info('finish waiting')
                    tasks.clear()

            except Exception as e:
                logger.error(doc['reference'] + ' : ' + str(e))
