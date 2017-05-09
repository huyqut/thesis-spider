from twitter_dev import TwitterDev


class Spider:

    @staticmethod
    def crawl_feeds(dev: TwitterDev = None):
        if dev is None:
            return
        friends = dev.api.GetFriendIDs()['ids']
        for fid in friends:
            print(fid)
