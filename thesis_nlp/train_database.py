import pymongo


def train_database(host: str = 'localhost', port: int = 27017):
    return pymongo.MongoClient(host, port)['train-database']


def train_collection():
    return train_database()['train-collection']
