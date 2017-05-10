from twitter_dev import TwitterDev
import re
import geograpy


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
            url_match = re.search("(?P<url>https?://[^\s]+)", message)
            if url_match is None:
                continue
            url = url_match.group(0)
            if len(url) < 23:
                continue
            print(url)
            print(geograpy.get_place_context(url).places)
