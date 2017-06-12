from twitter_dev import TwitterDev
import re
import time
import geograpy
import threading
import database
import thesis_logging


def crawl_feeds(dev: TwitterDev):
    logger = thesis_logging.get_logger('crawler')
    try:
        if dev is None:
            logger.error('There is no Twitter developer account detected.')
            return
        news_collection = database.news_collection()
        logger.info('ok')
        user_id = dev.api.VerifyCredentials()
        logger.info('Twitter Auth: ' + str(user_id.AsJsonString()))
        friends = dev.api.GetFriendIDs(user_id, stringify_ids = True)
        logger.info('Friends: ' + str(friends))
        logger.info('Start crawling')
        for status in dev.api.GetStreamFilter(follow = friends):
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
    except Exception as e:
        logger.error(e)


def locate_feeds(latest: int = 0):
    logger = thesis_logging.get_logger('locator')
    news_collection = database.news_collection()

    def parse_pages(id, url, collection):
        try:
            geo = geograpy.get_place_context(url)
            news_collection.update_one({'id': id},
                                       {'$set': {'places': geo.places,
                                                 'people': geo.people,
                                                 'organs': geo.organs}})
            if len(geo.places) > 0:
                for place in geo.places:
                    collection.update_one({'place': place}, {'$inc': {'count': 1}}, upsert = True)
        except Exception as e:
            logger.error(str(e))
    location_collection = database.location_collection()
    while True:
        documents = news_collection.find({'created_at': {'$gt': latest}})
        logger.info('Found ' + str(documents.count()) + ' after ' + str(latest))
        if documents.count() == 0:
            logger.warn('Nap and back in 5 seconds')
            time.sleep(5000)
            continue
        tasks = []
        logger.info('Start Locating')
        for doc in documents:
            try:
                ref = doc['reference']
                thread = threading.Thread(target = parse_pages, args = (doc['id'], ref, location_collection))
                tasks.append(thread)
                thread.start()
                if len(tasks) == 10:
                    for task in tasks:
                        task.join()
                    latest = doc['created_at']
                    tasks.clear()
            except Exception as e:
                logger.error(doc['reference'] + ' : ' + str(e))
