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
            link_list = ['cnn.it', 'nyti.ms', 'nbcnews', 'apne.ws', 'reut.rs', 'wapo.st', 'abcn.ws',
                         'ti.me', 'cbsn.ws', 'huffingtonpost.com', 'cnb.cx',
                         'huffp.st', 'forbes.com', 'telegraph.co', 'cnn.com', 'trib.al',
                         'express.co', 'gu.com', 'bloom.bg', 'hill.cm', 'natgeo.com',
                         'pbs.org', 'washingtonpost']
            ignore_list = ['bit.ly', 'twitter', 'tinyurl', 'goo.gl', 'facebook.com', ]
            dupliate_urls = {}
            for status in dev.api.GetStreamFilter(follow=friends):
                urls = status['entities']['urls']
                if len(urls) == 0:
                    continue
                url = urls[0]['expanded_url']

                if url is None:
                    continue

                if not any(x in url for x in link_list):
                    logger.info('Skip link ' + url)
                    continue

                if news_collection.find({'reference': url}).count() > 0:
                    logger.info('Skip duplicated ' + url)
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
                if "twitter" in article.canonical_link:
                    logger.info('delete ' + article.canonical_link)
                    news_collection.remove({"id": self.tweet_id})
                    return
                logger.info(
                    'Title for ' + article.top_image + '  -  ' + article.canonical_link + '\n' + article.title + '\n\n')
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
                                                     'vector': vector,
                                                     'title': article.title,
                                                     'text': article.text,
                                                     'image': article.top_image}})
                for place in geography_extractor.places:
                    self.collection.update_one({'place': place},
                                               {'$inc': {'count': 1}},
                                               upsert=True)

            except Exception as e:
                logger.error(str(e))

    location_collection = database.location_collection()
    duplicate_urls = {}
    tasks = []
    while True:
        documents = news_collection.find({'created_at': {'$gte': latest}}).limit(100)
        logger.info('Found ' + str(documents.count()) + ' after ' + str(latest))

        # Clean up remaining tasks
        if len(tasks) != 0:
            logger.info('Cleaning up remaining tasks')
            for task in tasks:
                task.join()
            tasks.clear()

        if documents.count() == 1:
            if crawler_finish:
                break
            logger.warn('Nap and back in 120 seconds')
            time.sleep(120)
            continue

        logger.info('Start Locating')
        index = 0

        for doc in documents:
            try:
                ref = doc['reference']
                latest = doc['created_at']
                image = doc.get('image')

                if latest >= 1498253429:
                    return

                if image is not None:
                    logger.info('image skip')
                    continue
                if news_collection.find({'reference': ref}).count() > 1:
                    logger.info('delete duplicate ' + ref)
                    news_collection.remove({"id": doc['id']})
                    continue

                thread = PageParser(doc['id'], ref, location_collection)
                tasks.append(thread)
                thread.start()
                time.sleep(5)
                index += 1
                if index % 5 == 0:
                    for task in tasks:
                        task.join()
                    tasks.clear()


            except Exception as e:
                logger.error(doc['reference'] + ' : ' + str(e))
