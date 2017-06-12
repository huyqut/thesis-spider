from twitter_dev import *
from spider import *
import warnings
import thesis_logging
from threading import Thread


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
    crawler = Thread(target = crawl_feeds, args = (twitter_dev,))
    locator = Thread(target = locate_feeds, args = ())
    crawler.start()
    locator.start()
    crawler.join()
    locator.join()
except Exception as e:
    logger.error('Finish thesis: ' + str(e))
    exit(2)
