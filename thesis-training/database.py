import pymongo


def news_database(host: str = 'localhost', port: int = 27017):
    return pymongo.MongoClient(host, port)['news-database']


def news_collection():
    return news_database()['news-collection']
