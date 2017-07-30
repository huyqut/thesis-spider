from os import path
from random import shuffle

import nltk
from gensim.corpora import WikiCorpus
from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from nltk.corpus import reuters
from nltk.tokenize import RegexpTokenizer
from nltk.stem.porter import PorterStemmer
import numpy as np
import logging
import os.path
import six
import sys
import thesis_nlp.train_database as train_database
import database as news_database
import thesis_logging
from bson.objectid import ObjectId

# Clustering
import hdbscan
from sklearn.datasets import make_blobs
from scipy.cluster.hierarchy import fcluster
from sklearn import preprocessing
from sklearn.metrics import pairwise_distances
import pickle
from sklearn.cluster import KMeans, Birch, AgglomerativeClustering
from collections import Counter
from sklearn.neighbors import NearestNeighbors

import math

# Plotting
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

program = os.path.basename(sys.argv[0])
logger = logging.getLogger(program)

logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
logging.root.setLevel(level=logging.INFO)
logger.info("running %s" % ' '.join(sys.argv))

google_news_word2vec_model_location = 'data/GoogleNews-vectors-negative300.bin.gz'
enwiki_bin_location = 'training/metawiki-20170401-pages-articles.xml.bz2'
enwiki_txt_location = 'training/wiki-documents.txt'
doc2vec_model_location = 'model/doc2vec-model-300.bin'
word2vec_model_location = 'model/word2vec-model.bin'
doc2vec_vectors_location = 'model/doc2vec-vectors.bin'
clustering_model_location = 'model/clustering_model.bin'
doc2vec_dimensions = 300
classifier_model_location = 'model/classifier-model.bin'

train_collection = train_database.train_collection()
news_collection = news_database.news_collection()


# Build the word2vec model from the corpus
# doc2vec.build_vocab(taggedDocuments)

def build_wiki_text():
    i = 0
    output = open(enwiki_txt_location, 'w+', encoding="utf-8")
    wiki = WikiCorpus(enwiki_bin_location, lemmatize=False, dictionary={})
    for text in wiki.get_texts():
        output.write(b' '.join(text).decode('utf-8') + '\n')
        i = i + 1
        if (i % 10000 == 0):
            logger.info("Saved " + str(i) + " articles")

    output.close()
    logger.info("Finished Saved " + str(i) + " articles")


def build_wiki_tagged():
    for idx, doc in enumerate(open(enwiki_txt_location, 'r', encoding="utf-8")):
        if idx < 1000000:
            yield TaggedDocument(words=nltk.word_tokenize(doc.lower()), tags=['tag_' + str(idx)])
        else:
            break


def build_news_tagged():
    logger = thesis_logging.get_logger('preprocess')
    latest = ObjectId("5950c589f296532a806e4f31")
    count = 0
    index = 1

    news_collection = train_collection
    duplicated_doc = {}
    while True:
        documents = news_collection.find({'_id': {'$gt': latest}})
        if documents.count() == 0:
            break
        for doc in documents:
            count += 1
            try:
                latest = doc['_id']
                if not doc.get('text'):
                    # print('Ignore', 'Count ' + str(count), 'Id ' + str(doc['id']), str(doc['created_at']),
                    #      doc['reference'])
                    continue
                content = doc['text']
                if len(content) < 100:
                    # logger.info('Ignore small content, Count ' + str(count))
                    continue
                if content not in duplicated_doc:
                    duplicated_doc[content] = True
                    index += 1
                    # logger.info(nltk.word_tokenize(content.lower()))
                    yield TaggedDocument(words=nltk.word_tokenize(content.lower()), tags=[doc['id']])

            except Exception as e:
                logger.error(doc['reference'] + ' : ' + str(e))


def train():
    # build_wiki_text()

    logger.info("Build TaggedDocuments from training docs")

    model = Doc2Vec(size=doc2vec_dimensions, iter=1, window=10, dbow_words=0,
                    seed=1337, min_count=5, workers=10, alpha=0.025, min_alpha=0.025)
    logger.info("Build Vocabulary from docs")
    model.build_vocab(build_wiki_tagged())
    for epoch in range(10):
        logger.info("epoch " + str(epoch))
        model.train(build_wiki_tagged(), total_examples=model.corpus_count, epochs=1)
        model.save(doc2vec_model_location)
        model.alpha -= 0.002  # decrease the learning rate
        model.min_alpha = model.alpha  # fix the learning rate, no decay


def load():
    # from gensim.models import Doc2Vec
    # doc2vec_model_location = 'model/doc2vec-model.bin'
    # model = Doc2Vec.load(doc2vec_model_location)
    # model.docvecs['']  # get vector from tag or index
    # model.docvecs.most_similar(positive=[model.docvecs['']]) # get most similar vector 1
    # model.docvecs.most_similar(positive=[model.infer_vector(words)]) # get most similar vector 2

    doc2vec_model_location = 'temp/doc2vec-model-300.bin'
    # doc2vec_model_location = 'D:/thesis-spider/thesis_nlp/temp/doc2vec-model-wiki-300.bin'
    logger.info("Loading model...")
    model = Doc2Vec.load(doc2vec_model_location)
    model.delete_temporary_training_data(keep_doctags_vectors=True, keep_inference=True)
    testIds = ['731550159583088641', '549916776734416896', '543030544981032961', '812672127891636224']

    # for id in testIds:
    #     list_similar = model.docvecs.most_similar(positive=[model.docvecs[id]])
    #     for similar in list_similar:
    #         print(similar)
    #         doc = train_collection.find_one({'id': similar[0]})
    #         print(doc['title'])
    #     print('\n')

    # cluster(model=model)
    for i in range(0, 5):
        evaluate(model)
        print('\n\n')


def get_ner_list(ner_list, doc):
    if len(doc['people']) == 0:
        return
    c = Counter(doc['people'])
    ner = c.most_common(1)[0]

    if ner not in ner_list:
        ner_list[ner] = []
    else:
        ner_list[ner].append(doc['id'])


def plot_data(data_vecs, clusterer):
    pca = PCA(n_components=50)
    fiftyDimVecs = pca.fit_transform(data_vecs)

    tsne = TSNE(n_components=2)
    twoDimVecs = tsne.fit_transform(fiftyDimVecs)

    # centers = pca.fit_transform(clusterer.cluster_centers_)
    # centerDimVecs = tsne.fit_transform(centers)
    # print(centerDimVecs)

    fig, ax = plt.subplots()
    idx = 0

    # for centerDimVec in centerDimVecs:
    #    ax.scatter(centerDimVec[0], centerDimVec[1], c='b')

    colors = plt.cm.get_cmap('hsv', len(clusterer.labels_))

    for twoDimVec in twoDimVecs:
        ax.scatter(twoDimVec[0], twoDimVec[1], c=colors(clusterer.labels_[idx]), s=0.1)
        idx += 1

    plt.show()


def cluster(model):
    idx = 0
    data_vecs = []
    data_tags = []
    for key in model.docvecs.doctags:
        if idx >= 1000:
            break
        data_vecs.append(model.docvecs[key])
        data_tags.append(key)
        idx += 1

    print('Infer doc vectors')
    docs = news_collection.find({})
    duplicate_doc = {}
    people_list = {}
    # for doc in docs:
    #     text = doc['text']
    #     if text in duplicate_doc:
    #         continue
    #     else:
    #         duplicate_doc[text] = True
    #
    #     data_vecs.append(model.infer_vector(doc['text']))
    #     data_tags.append(doc['id'])

    clusterer = clustering(doc_vecs=data_vecs)

    pickle.dump(clusterer, open(clustering_model_location, 'wb'))
    logger.info('Saved clustering model')
    idx = 0
    clusters_list = {}
    for label in clusterer.labels_:
        if label not in clusters_list:
            clusters_list[label] = []
        else:
            clusters_list[label].append(data_tags[idx])
        idx += 1
    print(len(clusters_list))

    for label in clusters_list:
        print(label)
        print(len(clusters_list[label]))
        for article in clusters_list[label]:
            print(article)
            doc = train_collection.find_one({'id': article})
            print(doc['title'])
        print('\n')

        # print('Inertia: ', clusterer.inertia_)
        # plot_data(data_vecs, clusterer)


def clustering(doc_vecs):
    def sqrt(list):
        return [math.sqrt(math.fabs(i)) * math.fabs(i) / i for i in list]

    print('Normalize vectors...')
    data_vecs = preprocessing.normalize(doc_vecs, norm='l2')

    vecs = []
    for vec in data_vecs:
        vec = sqrt(vec)
        vecs.append(vec)
        # print(max(vec), min(vec))

    data_vecs = preprocessing.normalize(vecs, norm='l2')
    print(data_vecs)

    # pca = PCA(n_components=15)
    # data_vecs = pca.fit_transform(data_vecs)

    print('Start clustering...')
    # distance = pairwise_distances(data_vecs, metric='l2')
    # clusterer = Birch(branching_factor=50, n_clusters=5, threshold=5, compute_labels = True)
    # clusterer = hdbscan.HDBSCAN(metric="euclidean", min_cluster_size=5, prediction_data=True)
    clusterer = KMeans(n_clusters=5, max_iter=1000)
    # clusterer =  AgglomerativeClustering(n_clusters=5, linkage="complete", affinity='euclidean')

    clusterer.fit(data_vecs)

    # print('Generate prediction data')
    # clusterer.generate_prediction_data()
    return clusterer


def evaluate(model):
    from pathlib import Path

    class ClusterData:
        def __init__(self, type, path, vec):
            self.type = type
            self.path = path
            self.vec = vec

    p = Path('./bbc')
    doc_vecs = []
    cluster_datas = []
    # List comprehension
    for f in p.iterdir():
        if f.is_dir():
            print(str(f))
            list_docs = Path(str(f))
            for doc_path in list_docs.iterdir():
                cluster_datas.append(ClusterData(f.name, doc_path, 0))
            print(len(cluster_datas))

    print('Infer vectors...')
    idx = 0
    for cluster_data in cluster_datas:
        with open(cluster_data.path, 'r') as doc_file:
            data = doc_file.read()
            vec = model.infer_vector(nltk.word_tokenize(data))
            doc_vecs.append(vec)
            cluster_datas[idx].vec = vec
            idx += 1

    clusterer = clustering(doc_vecs)

    neigh = NearestNeighbors(2)
    neigh.fit(doc_vecs)

    idx = 0
    clusters_list = {}
    clusters_list_path = {}
    for label in clusterer.labels_:
        if label not in clusters_list:
            clusters_list[label] = []
        clusters_list[label].append(cluster_datas[idx])
        idx += 1
    print(len(clusters_list))

    cluster_sum = 0
    doc_sum = 0
    for label in clusters_list:
        dict_docs = {}
        dict_doc_datas = {}
        idx = 0
        for article in clusters_list[label]:
            if article.type not in dict_docs:
                dict_docs[article.type] = 0
                dict_doc_datas[article.type] = []
            dict_docs[article.type] += 1
            dict_doc_datas[article.type].append(article)
            doc_sum += 1
            idx += 1

        max_key = max(dict_docs.keys(), key=(lambda key: dict_docs[key]))
        cluster_sum += dict_docs[max_key]
        print('Cluster: ', max_key)
        print('Cluster size:', len(clusters_list[label]))
        print(dict_docs)
        for key in dict_docs:
            if key != max_key:
                print(key, dict_doc_datas[key][0].path)
                print(neigh.kneighbors([dict_doc_datas[key][0].vec], 2))
        print('\n')
    print('Sum of cluster = ', cluster_sum, '/', doc_sum)
    print('Percentage = ', cluster_sum / doc_sum)


def eblow(data, n):
    from scipy.spatial.distance import cdist, pdist
    df = np.array(data)
    kMeansVar = [KMeans(n_clusters=k).fit(df) for k in range(2, n)]
    centroids = [X.cluster_centers_ for X in kMeansVar]
    k_euclid = [cdist(df, cent) for cent in centroids]
    dist = [np.min(ke, axis=1) for ke in k_euclid]
    wcss = [sum(d ** 2) for d in dist]
    tss = sum(pdist(df) ** 2) / df.shape[0]
    bss = tss - wcss
    plt.plot(bss)
    plt.show()


if __name__ == '__main__':
    # train()
    load()
