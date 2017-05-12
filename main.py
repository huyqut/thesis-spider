from twitter_dev import *
from spider import *
import warnings

warnings.filterwarnings("ignore", category = DeprecationWarning)

if not os.path.exists(DATA_FOLDER):
    try:
        if not TwitterDev.prompt_init():
            exit()
    except Exception as e:
        print('Error: ' + str(e))
        exit(1)

username = input('Username: ')
try:
    twitterDev = TwitterDev(DATA_FOLDER + '/' + username)
    spider = Spider(twitterDev)
    #spider.crawl_feeds()
    spider.locate_feeds()
except Exception as e:
    print('Error: ' + str(e))
    exit(2)
