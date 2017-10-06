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
import random
# Evaluate
from pathlib import Path
import csv
import math
from collections import Counter
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KNeighborsClassifier
import re

# Plotting
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

from thesis_nlp.train import ClusterData


# Write wiki files
def write_wiki_files():
    idx = 0
    file_wiki = './datasets/wiki/wiki-documents-drop.txt'
    docs_list = []
    path_file = Path('./datasets/wiki-documents.txt')
    with open(path_file, 'r', encoding="utf-8") as doc_file:
        for doc in doc_file:
            docs_list += chunk_data(doc)
            print(len(docs_list))
            if len(docs_list) >= 400000:
                break
    with open(file_wiki, 'w', encoding="utf-8") as doc_file:
        for doc in docs_list:
            doc_file.write(doc)
            idx += 1
            if idx >= 400000:
                break
            doc_file.write('\n')


def write_web_news():
    logger = thesis_logging.get_logger('preprocess')

    link_list = ['cnn.it', 'nyti.ms', 'nbcnews', 'apne.ws', 'reut.rs', 'wapo.st',
                 'abcn.ws', 'nbcbay.com', 'bbc.in', 'huff.to',
                 'ti.me', 'cbsn.ws', 'huffingtonpost.com', 'cnb.cx', 'cnnmon.ie',
                 'huffp.st', 'forbes.com', 'telegraph.co', 'cnn.com', 'trib.al',
                 'express.co', 'gu.com', 'bloom.bg', 'hill.cm', 'natgeo.com',
                 'pbs.org', 'washingtonpost', 'news.sky.com']
    for source in link_list:
        # latest = ObjectId("59abbfedf296532f80d18a47")  # dyncorp
        # latest = ObjectId("59abc7e2f296532ad483f4b6")  # lds
        # latest = ObjectId("59acc20df296533c88dbaed6")  # tm
        latest = ObjectId("5942946efe43ad1da80b1a79")  # news
        index = 0
        path_file = './datasets/insensitive/news/' + source.replace('.', '_') + '_'
        train_collection = news_database.news_collection()
        duplicated_doc = {}
        while True:
            documents = train_collection.find({'_id': {'$gt': latest}, 'reference': {'$regex': '.*' + source + '.*'}})
            if documents.count() == 0:
                break
            for doc in documents:
                try:
                    latest = doc['_id']
                    if not doc.get('text'):
                        # print('Ignore', 'Count ' + str(count), 'Id ' + str(doc['id']), str(doc['created_at']),
                        #      doc['reference'])
                        continue
                    content = doc['text']
                    if len(content) < 1000:
                        # logger.info('Ignore small content, Count ' + str(count))
                        continue
                    title = doc['title']
                    if len(title) > 60:
                        title = title[0:60]
                    title = "".join(x for x in title if x.isalnum())
                    if content not in duplicated_doc:
                        duplicated_doc[content] = True
                        index += 1
                        # logger.info(nltk.word_tokenize(content.lower()))
                        with open(path_file + title + '.txt', 'w', encoding="utf-8") as doc_file:
                            doc_file.write(doc['text'])

                except Exception as e:
                    logger.error(doc['reference'] + ' : ' + str(e))
        print(source, index)


def split_data():
    path_file = './datasets/sensitive/tm/Associate_Agreement.txt'
    path_saved_file = './datasets/sensitive/tm/'
    file_name = 0
    with open(path_file, 'r') as doc_file:
        data = doc_file.read()
        docs = []
        chunks = len(data)
        chunk_size = CHAR_LEN
        i = 0
        while i < chunks:
            lim = i + chunk_size
            while lim < chunks and data[lim] != ' ':
                lim += 1
            if chunks - lim < chunk_size:
                doc = data[i:chunks]
                i = chunks
            else:
                doc = data[i:lim]
                i = lim
            docs.append(doc)
            with open(str(path_saved_file) + 'Associate_Agreement_' + str(file_name) + '.txt',
                      'w') as new_file:
                new_file.write(doc)
            file_name += 1


def chunk_data(data, size=-1):
    docs = []
    chunks = len(data)
    chunk_size = CHAR_LEN if size == -1 else size
    i = 0
    while i < chunks:
        lim = i + chunk_size
        while lim < chunks and data[lim] != ' ':
            lim += 1
        if chunks - lim < chunk_size:
            doc = data[i:chunks]
            i = chunks
        else:
            doc = data[i:lim]
            i = lim
        docs.append(doc)
    return docs


def chunk_first_data(data):
    docs = []

    # For full content
    docs.append(data)
    return docs

    chunks = len(data)
    chunk_size = CHAR_LEN
    i = 0
    while i < chunks:
        lim = i + chunk_size
        while lim < chunks and data[lim] != ' ':
            lim += 1
        if chunks - lim < chunk_size:
            doc = data[i:chunks]
            i = chunks
        else:
            doc = data[i:lim]
            i = lim
        docs.append(doc)
        break
    return docs


def chunk_multi_data(data):
    docs = []

    docs.append(data)
    return docs

    chunks = len(data)
    try:
        ran_list = random.sample(range(0, chunks - CHAR_LEN), 20)
    except ValueError:
        docs.append(data)
        return docs

    for ran in ran_list:
        docs.append(data[ran:ran + CHAR_LEN])
    return docs


def get_data_mormon():
    print('Get moron')
    path_file = './datasets/sensitive/mormon-handbook-of-instructions-1999.txt'
    cluster_datas = []
    list_docs = []
    with open(path_file, 'r') as doc_file:
        data = doc_file.read()
        docs = chunk_data(data, 2000)
    for doc in docs:
        list_docs += chunk_multi_data(doc)
    random.shuffle(list_docs)
    size = len(list_docs) / 2
    print(size * 2)
    for doc in list_docs:
        if size < 1:
            cluster_datas.append(ClusterData('sen', 'private-e', 0, doc, 'test'))
        else:
            cluster_datas.append(ClusterData('sen', 'private-e', 0, doc, 'train'))
            size -= 1

    return cluster_datas


def get_data_snowden():
    print('Get snowden')
    folder_path = Path('./datasets/sensitive/snowden')
    cluster_datas = []
    for p in folder_path.iterdir():
        list_docs_path = [x for x in p.iterdir()]
        list_docs = []
        for doc_path in list_docs_path:
            with open(doc_path, 'r', encoding="utf-8") as doc_file:
                data = doc_file.read()
                list_docs += chunk_multi_data(data)
        random.shuffle(list_docs)
        size = len(list_docs) / 2
        for doc in list_docs:
            if size < 1:
                cluster_datas.append(ClusterData('sen', 'private-e', 0, doc, 'test'))
            else:
                cluster_datas.append(ClusterData('sen', 'private-e', 0, doc, 'train'))
                size -= 1
    print('len:', len(cluster_datas))
    return cluster_datas


def get_data_enron(max_size):
    print('Get enron')
    path_file = './datasets/sensitive/enron_mail_20150507.txt'
    cluster_datas = []
    with open(path_file, 'r') as doc_file:
        data = doc_file.read()
        docs = chunk_data(data)
        random.shuffle(docs)
        docs = docs[0:max_size] if max_size != -1 else docs
        size = len(docs) / 2
        print('len:', size * 2)
        for doc in docs:
            if size < 1:
                cluster_datas.append(ClusterData('sen', 'private-e', 0, doc, 'test'))
            else:
                cluster_datas.append(ClusterData('sen', 'private-e', 0, doc, 'train'))
                size -= 1
    return cluster_datas


def get_wiki(wiki_size):
    print('Get wiki')
    path_file = Path('./datasets/wiki/wiki-documents.txt')
    list_docs = []
    idx = 0
    with open(path_file, 'r', encoding="utf-8") as doc_file:
        for doc in doc_file:
            if len(doc) < CHAR_LEN:
                continue
            if idx >= wiki_size:
                break
            docs = chunk_first_data(doc)
            list_docs += docs
            idx += len(docs)
    cluster_datas = []
    random.shuffle(list_docs)
    size = len(list_docs) / 2
    print('len:', size * 2)
    for doc in list_docs:
        if size < 1:
            cluster_datas.append(ClusterData('insen', 'ne', 0, doc, 'test'))
        else:
            cluster_datas.append(ClusterData('insen', 'ne', 0, doc, 'train'))
            size -= 1
    return cluster_datas


def get_sensitive(folder):
    print('Get sensitive ' + folder)
    p = Path('./datasets/sensitive/' + folder)
    cluster_datas = []
    list_docs_path = [x for x in p.iterdir()]
    list_docs = []
    for doc_path in list_docs_path:
        with open(doc_path, 'r', encoding="utf-8") as doc_file:
            data = doc_file.read()
            list_docs += chunk_multi_data(data)
    random.shuffle(list_docs)
    size = len(list_docs) / 2
    print(str(p), size * 2)
    for data in list_docs:
        if size < 1:
            cluster_datas.append(ClusterData('sen', 'private-e', 0, data, 'test'))
        else:
            cluster_datas.append(ClusterData('sen', 'private-e', 0, data, 'train'))
            size -= 1
    return cluster_datas


def get_insensitive(folder):
    print('Get insensitive ' + folder)
    p = Path('./datasets/insensitive/' + folder)
    cluster_datas = []
    list_docs_path = [x for x in p.iterdir()]
    list_docs = []
    for doc_path in list_docs_path:
        with open(doc_path, 'r', encoding="utf-8") as doc_file:
            data = doc_file.read()
            list_docs += chunk_first_data(data)
    random.shuffle(list_docs)
    size = len(list_docs) / 2
    print(str(p), size * 2)
    for data in list_docs:
        if size < 1:
            cluster_datas.append(ClusterData('insen', 'public-e', 0, data, 'test'))
        else:
            cluster_datas.append(ClusterData('insen', 'public-e', 0, data, 'train'))
            size -= 1
    return cluster_datas


def load_model():
    doc2vec_model_location = 'temp/doc2vec-model-300.bin'
    # doc2vec_model_location = 'temp/doc2vec-model-400k-300.bin'
    # doc2vec_model_location = 'D:/thesis-spider/thesis_nlp/temp/doc2vec-model-wiki-300.bin'
    model = Doc2Vec.load(doc2vec_model_location)
    model.delete_temporary_training_data(keep_doctags_vectors=True, keep_inference=True)
    return model


def build_docvecs(model, cluster_datas):
    idx = 0
    doc_vecs = []
    labels = []
    for cluster_data in cluster_datas:
        words = nltk.word_tokenize(cluster_data.text.lower())
        vec = model.infer_vector(words, steps=STEP, alpha=ALPHA)
        doc_vecs.append(vec)
        labels.append(cluster_data.type)
        cluster_datas[idx].vec = vec
        idx += 1
    return doc_vecs, labels, cluster_datas


NEIGHBOR = 7
STEP = 3
ALPHA = 0.2

CHAR_LEN = 5000


def evaluate_dataset(model, cluster_datas):
    import datetime
    print('Evaluate with model, k =', NEIGHBOR, 'step =', STEP, 'alpha =', ALPHA, 'CHAR LEN =', CHAR_LEN)
    for i in range(0, 1):
        train_set = [cluster_data for cluster_data in cluster_datas if cluster_data.split == 'train']
        test_set = [cluster_data for cluster_data in cluster_datas if cluster_data.split == 'test']
        print('Build docvecs for train set', len(train_set), datetime.datetime.now().time())
        doc_vecs_train, labels_train, cluster_datas_train = build_docvecs(model, train_set)

        print('Normalize train vectors...', datetime.datetime.now().time())
        doc_vecs_train = preprocessing.normalize(doc_vecs_train, norm='l2')

        print('Training data...', len(labels_train), datetime.datetime.now().time())
        neigh = KNeighborsClassifier(n_neighbors=NEIGHBOR)
        neigh.fit(doc_vecs_train, labels_train)

        print('Build docvecs for test set', len(test_set))
        doc_vecs_test, labels_test, cluster_datas_test = build_docvecs(model, test_set)

        print('Normalize test vectors...', datetime.datetime.now().time())
        doc_vecs_test = preprocessing.normalize(doc_vecs_test, norm='l2')

        print('Testing...', len(labels_test), datetime.datetime.now().time())
        result = neigh.predict(doc_vecs_test)

        # clusterer = KMeans(n_clusters=5, max_iter=1000, n_init=1000)
        # clusterer.fit_predict(doc_vecs_test, labels_test)
        # result = clusterer.labels_
        # print(labels_test)
        # print(result)

        

        corrects = sum((result) == labels_test)
        errors = len(cluster_datas_test) - corrects
        error_rate = float(errors) / len(cluster_datas_test)
        print('error', errors, '/', len(cluster_datas_test), datetime.datetime.now().time())
        print('error rate:', error_rate)
        print('Correct:', corrects / (len(cluster_datas_test)))

        idx = 0
        FP_public_e_num = 0.0
        TN_private_e_num = 0.0
        TP_public_e_num = 0.0
        FN_private_e_num = 0.0
        FP_public_ne_num = 0.0
        for cluster_data in cluster_datas_test:
            if cluster_data.path == 'private-e':
                if result[idx] != labels_test[idx]:
                    FN_private_e_num += 1
                else:
                    TN_private_e_num += 1
            elif cluster_data.path == 'public-e':
                if result[idx] != labels_test[idx]:
                    FP_public_e_num += 1
                else:
                    TP_public_e_num += 1
            else:
                if result[idx] != labels_test[idx]:
                    FP_public_ne_num += 1
                else:
                    TP_public_e_num += 1
            idx += 1

        print('Misclassify sensitive:', FN_private_e_num, '/', (TN_private_e_num + FN_private_e_num))

        print('FP (public e): ', FP_public_e_num, '/', (FP_public_e_num + TN_private_e_num), ': ',
              FP_public_e_num / (FP_public_e_num + TN_private_e_num))
        print('FN (private e):', FN_private_e_num, '/', (FN_private_e_num + TP_public_e_num), ': ',
              FN_private_e_num / (FN_private_e_num + TP_public_e_num))
        print('FP (public ne):', FP_public_ne_num, '/', (FP_public_ne_num + TN_private_e_num), ': ',
              FP_public_ne_num / (FP_public_ne_num + TN_private_e_num))

    print('\n\n')


def evaluate():
    model = load_model()

    # for i in range(0,5):
    #     sensitive_datas = get_sensitive('tm') + get_sensitive('dyncorp') + get_data_mormon() + get_data_snowden() + get_data_enron(10000)
    #     insensitive_datas = get_insensitive('tm') + get_insensitive('ldschurch') + get_insensitive('dyncorp') + get_insensitive('news') + get_wiki(10000)
    #     cluster_datas_all = sensitive_datas + insensitive_datas
    #     print('Sensitive:', len(sensitive_datas))
    #     print('InSensitive:', len(insensitive_datas))
    #     print('Total:', len(sensitive_datas) + len(insensitive_datas))
    #     evaluate_dataset(model, cluster_datas=cluster_datas_all)

    for i in range(0, 5):
        cluster_datas_mt = get_sensitive('tm') + get_wiki(10000) + get_insensitive('tm') + get_insensitive('news')
        evaluate_dataset(model, cluster_datas=cluster_datas_mt)

        cluster_datas_mormon = get_data_mormon() + get_wiki(10000) + get_insensitive('ldschurch') + get_insensitive(
            'news')
        evaluate_dataset(model, cluster_datas=cluster_datas_mormon)


        cluster_datas_enron = get_data_enron(10000) + get_wiki(10000) + get_insensitive('news')
        evaluate_dataset(model, cluster_datas=cluster_datas_enron)

        cluster_datas_dyn = get_sensitive('dyncorp') + get_wiki(10000) + get_insensitive('dyncorp') + get_insensitive(
            'news')
        evaluate_dataset(model, cluster_datas=cluster_datas_dyn)

        cluster_datas_snowden = get_data_snowden() + get_wiki(10000) + get_insensitive('news')
        evaluate_dataset(model, cluster_datas=cluster_datas_snowden)


        # print('\n *************** \n Evaluate all datasets \n')


evaluate()
