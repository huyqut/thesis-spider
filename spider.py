from twitter_dev import TwitterDev
import re
import time
import geograpy
import database
import thesis_logging
import numpy as np
from thesis_nlp.convert import NewsConverter
from threading import Thread
from newspaper import Article

crawler_finish = False


def crawl_feeds(dev: TwitterDev, duration: int = 0):
    global crawler_finish
    logger = thesis_logging.get_logger('crawler')
    while True:
        try:
            if dev is None:
                logger.error('There is no Twitter developer account detected.')
                return
            news_collection = database.news_collection()
            logger.info('ok')
            user_id = dev.api.VerifyCredentials()
            logger.info('Twitter Auth: ' + str(user_id.AsJsonString()))
            friends = dev.api.GetFriendIDs(user_id, stringify_ids=True)
            logger.info('Friends: ' + str(friends))
            logger.info('Start crawling')
            start = int(round(time.time()) * 1000)
            for status in dev.api.GetStreamFilter(follow=friends):
                message = status['text']
                url_match = re.search("(?P<url>https?://[^\s]+)", message)
                if url_match is None:
                    continue
                url = url_match.group(0)
                if len(url) < 23:
                    continue

                timestamp = int(time.mktime(time.strptime(status['created_at'], '%a %b %d %H:%M:%S +0000 %Y')))

                document = {
                    'id': status['id'],
                    'created_at': timestamp,
                    'reference': url
                }
                news_collection.insert_one(document)
                logger.info('Insert ' + url + '  created at ' + str(timestamp))
                # if duration != 0 and int(round(time.time()) * 1000) - start > duration:
                #    break
        except Exception as e:
            logger.error(e)
        finally:
            # crawler_finish = True
            logger.info('Finish crawling')
            logger.info('Sleeping 5s to start again...')
            time.sleep(5)


def locate_feeds(news_converter: NewsConverter, latest: int = 0, ):
    global crawler_finish
    logger = thesis_logging.get_logger('locator')
    news_collection = database.news_collection()

    class VectorConverter(Thread):
        def __init__(self, text):
            super().__init__()
            self.text = text
            self.vector = []

        def run(self):
            self.vector = news_converter.convert_doc_to_vector(self.text).tolist()

    class GeographyExtractor(Thread):
        def __init__(self, text):
            super().__init__()
            self.text = text
            self.places = []
            self.people = []
            self.organs = []

        def run(self):
            context = geograpy.get_place_context(text=self.text)
            self.places = context.places
            self.people = context.people
            self.organs = context.organs

    class PageParser(Thread):
        def __init__(self, tweet_id, url, collection):
            super().__init__()
            self.tweet_id = tweet_id
            self.url = url
            self.collection = collection

        def run(self):
            try:

                logger.info('Parse ' + self.url)
                article = Article(self.url)
                article.download()
                article.parse()
                logger.info('Description for ' + self.url + '\n' + article.meta_description + '\n\n')
                logger.info('Latest: ' + str(latest))
                vector_converter = VectorConverter(article.text)
                geography_extractor = GeographyExtractor(article.text)
                vector_converter.start()
                geography_extractor.start()
                geography_extractor.join()
                vector_converter.join()
                vector = vector_converter.vector
                news_collection.update_one({'id': self.tweet_id},
                                           {'$set': {'places': geography_extractor.places,
                                                     'people': geography_extractor.people,
                                                     'organs': geography_extractor.organs,
                                                     'vector': vector}})
                for place in geography_extractor.places:
                    self.collection.update_one({'place': place},
                                               {'$inc': {'count': 1}},
                                               upsert=True)

            except Exception as e:
                logger.error(str(e))

    location_collection = database.location_collection()
    while True:
        documents = news_collection.find({'created_at': {'$gte': latest}}).limit(50)
        logger.info('Found ' + str(documents.count()) + ' after ' + str(latest))
        if documents.count() == 0:
            if crawler_finish:
                break
            logger.warn('Nap and back in 5 seconds')
            time.sleep(5)
            continue
        tasks = []
        logger.info('Start Locating')
        index = 0
        for doc in documents:
            try:
                ref = doc['reference']
                thread = PageParser(doc['id'], ref, location_collection)
                tasks.append(thread)
                thread.start()
                index += 1
                if index % 10 == 0 or index == documents.count():
                    for task in tasks:
                        task.join()
                    latest = doc['created_at']
                    tasks.clear()
            except Exception as e:
                logger.error(doc['reference'] + ' : ' + str(e))
