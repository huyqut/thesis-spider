from gensim.models.doc2vec import TaggedDocument

import database
import thesis_logging
from threading import Thread
from newspaper import Article
import nltk
import newspaper

def build_tagged():
    logger = thesis_logging.get_logger('preprocess')
    latest = 0
    count = 0
    index = 1

    news_collection = database.news_collection()
    duplicated_doc = {}
    while True:
        documents = news_collection.find({'created_at': {'$gte': latest}})
        if documents.count() == 0:
            break
        for doc in documents:
            count += 1
            try:
                latest = doc['created_at']
                if not doc.get('text'):
                    print('Ignore', 'Count ' + str(count), 'Id ' + str(doc['id']), str(doc['created_at']), doc['reference'])
                    continue
                content = doc['text']
                if content not in duplicated_doc:
                    duplicated_doc[content] = True
                    index += 1
                    logger.info(nltk.word_tokenize(content.lower()))
                    yield TaggedDocument(words=nltk.word_tokenize(content.lower()), tags=[index])

            except Exception as e:
                logger.error(doc['reference'] + ' : ' + str(e))



cnn_paper = newspaper.build('http://cnn.com')
print(len(cnn_paper.articles))
for article in cnn_paper.articles:
    print(article.url)

