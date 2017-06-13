from twitter_dev import *
from spider import *
import warnings
import thesis_logging
from threading import Thread
from thesis_nlp.convert import NewsConverter


warnings.filterwarnings("ignore", category = DeprecationWarning)
logger = thesis_logging.get_logger()

if not os.path.exists(DATA_FOLDER):
    try:
        if not TwitterDev.prompt_init():
            exit()
    except Exception as e:
        logger.error('Finish thesis: ' + str(e))
        exit(1)

username = input('Username: ')
try:
    logger.info('User ' + username + " requests authentication")
    twitter_dev = TwitterDev(DATA_FOLDER + '/' + username)
    news_converter = NewsConverter()
    crawler = Thread(target=crawl_feeds, args=(twitter_dev, 10000))
    locator = Thread(target=locate_feeds, args=(news_converter, int(round(time.time() * 1000))))
    crawler.start()
    locator.start()
    crawler.join()
    locator.join()
except Exception as e:
    logger.error('Finish thesis: ' + str(e))
    exit(2)
