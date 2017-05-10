from twitter_dev import TwitterDev
import re


class Spider:

    @staticmethod
    def crawl_feeds(dev: TwitterDev = None):
        if dev is None:
            return
        user_id = dev.api.VerifyCredentials().AsDict()['id']
        friends = dev.api.GetFriendIDs(user_id, stringify_ids = True)
        for status in dev.api.GetStreamFilter(follow = friends):
            if not status['truncated']:
                message = status['text']
            else:
                message = status['extended_tweet']['full_text']
            urls = re.search("(?P<url>https?://[^\s]+)", message)
            if urls is None:
                continue
            print(urls.group("url"))

