from twitter_dev import *
from spider import *
from geograpy import *

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
    Spider.crawl_feeds(twitterDev)
except Exception as e:
    print('Error: ' + str(e))
    exit(2)
